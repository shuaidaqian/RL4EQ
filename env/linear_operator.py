# -*- coding: utf-8 -*-
"""整帧块信道线性算子 H 与 H^H。"""

from __future__ import annotations

import os

import torch


class LinearChannelOperator:
    """把带 soft-tail 的有限长 CIR 表示为当前帧线性算子。"""

    def __init__(self, frame_len: int, max_delay: int, backend: str | None = None):
        self.frame_len = int(frame_len)
        self.max_delay = int(max_delay)
        self.backend = backend or os.environ.get("RL4EQ_LINEAR_OPERATOR_BACKEND", "direct")
        if self.backend not in {"direct", "fft"}:
            raise ValueError("线性算子 backend 必须为 direct 或 fft。")
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
        if self.backend == "fft":
            # 长度至少 N+L：线性卷积的循环混叠只落在被丢弃的前 L 个位置，
            # 因此截取 [L, L+N) 与逐 tap 的带历史尾部线性卷积等价。
            size = self._fft_size()
            convolved = torch.fft.ifft(
                torch.fft.fft(padded, n=size) * torch.fft.fft(cir_b, n=size), n=size,
            )
            output = convolved[:, self.max_delay:self.max_delay + self.frame_len]
            if not padded.is_complex() and not cir_b.is_complex():
                output = output.real
            return output if batched else output.squeeze(0)
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
        if self.backend == "fft":
            size = self._fft_size()
            correlated = torch.fft.ifft(
                torch.fft.fft(y_b, n=size) * torch.fft.fft(cir_b, n=size).conj(), n=size,
            )[:, :self.frame_len]
            if not y_b.is_complex() and not cir_b.is_complex():
                correlated = correlated.real
            return correlated if batched else correlated.squeeze(0)
        x_grad = torch.zeros_like(y_b)
        for delay in range(self.max_delay + 1):
            if delay == 0:
                shifted = y_b
            else:
                shifted = torch.zeros_like(y_b)
                shifted[:, :-delay] = y_b[:, delay:]
            x_grad = x_grad + torch.conj(cir_b[:, delay]).unsqueeze(1) * shifted
        return x_grad if batched else x_grad.squeeze(0)

    def _fft_size(self) -> int:
        return 1 << (self.frame_len + self.max_delay - 1).bit_length()

    def _validate(self, x: torch.Tensor, cir: torch.Tensor, tail: torch.Tensor) -> None:
        if x.shape[1] != self.frame_len:
            raise ValueError("x 的帧长不匹配。")
        if cir.shape[0] != x.shape[0] or cir.shape[1] != self.max_delay + 1:
            raise ValueError("cir 形状必须为 (B, max_delay + 1)。")
        if tail.shape != (x.shape[0], self.max_delay):
            raise ValueError("tail 形状必须为 (B, max_delay)。")


class CrossFrameLinearOperator:
    """把当前帧及下一帧已接收前缀表示为同一段连续线性观测。"""

    def __init__(
        self,
        frame_len: int,
        max_delay: int,
        lookahead_len: int,
        backend: str | None = None,
    ):
        self.frame_len = int(frame_len)
        self.max_delay = int(max_delay)
        self.lookahead_len = int(lookahead_len)
        if self.lookahead_len <= 0 or self.lookahead_len > self.max_delay:
            raise ValueError("lookahead_len 必须在 1 到 max_delay 之间。")
        self._operator = LinearChannelOperator(
            self.frame_len + self.lookahead_len,
            self.max_delay,
            backend=backend,
        )

    def forward(
        self,
        current_symbols: torch.Tensor,
        lookahead_symbols: torch.Tensor,
        cir: torch.Tensor,
        previous_tail: torch.Tensor,
    ) -> torch.Tensor:
        """计算当前帧和后续窗口的联合接收波形。"""

        current_b, current_unbatched = self._as_batch(current_symbols)
        lookahead_b, lookahead_unbatched = self._as_batch(lookahead_symbols)
        if current_unbatched != lookahead_unbatched:
            raise ValueError("当前帧与 lookahead 必须具有相同的 batch 维度。")
        if current_b.shape[1] != self.frame_len:
            raise ValueError("current_symbols 的帧长不匹配。")
        if lookahead_b.shape[1] != self.lookahead_len:
            raise ValueError("lookahead_symbols 的长度不匹配。")
        joined = torch.cat((current_b, lookahead_b), dim=1)
        result = self._operator.forward(joined, cir, previous_tail)
        return result if not current_unbatched else result.squeeze(0)

    def adjoint(
        self,
        observations: torch.Tensor,
        cir: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回联合观测对当前帧与后续前缀符号的伴随。"""

        values, unbatched = self._as_batch(observations)
        if values.shape[1] != self.frame_len + self.lookahead_len:
            raise ValueError("observations 的长度必须等于当前帧与 lookahead 之和。")
        gradient = self._operator.adjoint(values, cir)
        current, lookahead = gradient[:, :self.frame_len], gradient[:, self.frame_len:]
        if unbatched:
            return current.squeeze(0), lookahead.squeeze(0)
        return current, lookahead

    @staticmethod
    def _as_batch(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if values.ndim == 1:
            return values.unsqueeze(0), True
        if values.ndim == 2:
            return values, False
        raise ValueError("跨帧观测只支持一维或二维张量。")
