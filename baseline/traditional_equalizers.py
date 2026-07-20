# -*- coding: utf-8 -*-
"""基于 Adapt Pilot 的 RLS 判决反馈均衡基线。"""

import torch

from baseline.mmse_equalizer import _causal_windows, _to_complex


class DFERLSEqualizer:
    """使用 RLS 学习前向 FIR，并在 Data 段递归输出软判决。"""

    def __init__(self, forgetting_factor: float = 0.995, initial_inverse: float = 10.0):
        self.forgetting_factor = float(forgetting_factor)
        self.initial_inverse = float(initial_inverse)

    def equalize(
        self,
        rx_symbols: torch.Tensor,
        adapt_pilot_symbols: torch.Tensor,
        adapt_pilot_mask: torch.Tensor,
        max_delay_symbols: int,
        snr_db: float,
    ) -> torch.Tensor:
        del snr_db
        rx = _to_complex(rx_symbols)
        pilot_count = int(adapt_pilot_mask.sum().item())
        filter_len = max(3, min(int(max_delay_symbols) + 1, pilot_count))
        windows = _causal_windows(rx, filter_len)
        weights = torch.zeros(filter_len, dtype=torch.complex64)
        inverse = torch.eye(filter_len, dtype=torch.complex64) * self.initial_inverse
        lam = self.forgetting_factor
        for index in torch.nonzero(adapt_pilot_mask, as_tuple=False).flatten().tolist():
            vector = windows[index]
            desired = adapt_pilot_symbols[index].to(torch.complex64)
            projected = inverse @ vector
            denominator = lam + vector.conj() @ projected
            # 理论上该二次型为正实数；数值计算可能留下极小虚部。
            gain = projected / denominator.real.clamp_min(1e-8)
            error = desired - weights.conj() @ vector
            weights = weights + gain * error.conj()
            inverse = (inverse - torch.outer(gain, vector.conj()) @ inverse) / lam
        return (windows @ weights.conj()).real.float()
