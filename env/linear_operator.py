# -*- coding: utf-8 -*-
"""整帧块信道线性算子 H 与 H^H。"""

from __future__ import annotations

import torch


class LinearChannelOperator:
    """把带 soft-tail 的有限长 CIR 表示为当前帧线性算子。"""

    def __init__(self, frame_len: int, max_delay: int):
        self.frame_len = int(frame_len)
        self.max_delay = int(max_delay)
        if self.frame_len <= 0:
            raise ValueError("frame_len 必须为正。")
        if self.max_delay < 0:
            raise ValueError("max_delay 不能为负。")

    def forward(self, x: torch.Tensor, cir: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        batched = x.ndim == 2
        x_b = x if batched else x.unsqueeze(0)
        cir_b = cir if cir.ndim == 2 else cir.unsqueeze(0)
        tail_b = tail if tail.ndim == 2 else tail.unsqueeze(0)
        self._validate(x_b, cir_b, tail_b)
        padded = torch.cat([tail_b.to(x_b.device), x_b], dim=1)
        y = torch.zeros_like(x_b)
        indices = torch.arange(self.frame_len, device=x_b.device)
        origin = self.max_delay
        for delay in range(self.max_delay + 1):
            y = y + cir_b[:, delay].unsqueeze(1) * padded[:, origin + indices - delay]
        return y if batched else y.squeeze(0)

    def adjoint(self, y: torch.Tensor, cir: torch.Tensor) -> torch.Tensor:
        batched = y.ndim == 2
        y_b = y if batched else y.unsqueeze(0)
        cir_b = cir if cir.ndim == 2 else cir.unsqueeze(0)
        if y_b.shape[1] != self.frame_len:
            raise ValueError("y 的帧长不匹配。")
        if cir_b.shape[1] != self.max_delay + 1:
            raise ValueError("cir 长度必须等于 max_delay + 1。")
        x_grad = torch.zeros_like(y_b)
        for delay in range(self.max_delay + 1):
            if delay == 0:
                shifted = y_b
            else:
                shifted = torch.zeros_like(y_b)
                shifted[:, :-delay] = y_b[:, delay:]
            x_grad = x_grad + torch.conj(cir_b[:, delay]).unsqueeze(1) * shifted
        return x_grad if batched else x_grad.squeeze(0)

    def _validate(self, x: torch.Tensor, cir: torch.Tensor, tail: torch.Tensor) -> None:
        if x.shape[1] != self.frame_len:
            raise ValueError("x 的帧长不匹配。")
        if cir.shape[0] != x.shape[0] or cir.shape[1] != self.max_delay + 1:
            raise ValueError("cir 形状必须为 (B, max_delay + 1)。")
        if tail.shape != (x.shape[0], self.max_delay):
            raise ValueError("tail 形状必须为 (B, max_delay)。")
