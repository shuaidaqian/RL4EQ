# -*- coding: utf-8 -*-
"""只使用 Adapt Pilot 拟合的时域 LMMSE-FIR 均衡器。"""

import torch


def _to_complex(rx_symbols: torch.Tensor) -> torch.Tensor:
    return torch.complex(rx_symbols[:, 0].float(), rx_symbols[:, 1].float())


def _causal_windows(signal: torch.Tensor, filter_len: int) -> torch.Tensor:
    padded = torch.cat([torch.zeros(filter_len - 1, dtype=signal.dtype), signal])
    return torch.stack(
        [padded[index : index + filter_len].flip(0) for index in range(signal.numel())]
    )


class LMMSEFIREqualizer:
    """用正则化最小二乘直接学习接收窗口到发送符号的 FIR 映射。"""

    def __init__(self, regularization: float = 1e-2):
        self.regularization = float(regularization)

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
        filter_len = max(3, min(2 * int(max_delay_symbols) + 1, pilot_count))
        windows = _causal_windows(rx, filter_len)
        train_x = windows[adapt_pilot_mask]
        train_y = adapt_pilot_symbols[adapt_pilot_mask].to(torch.complex64)
        identity = torch.eye(filter_len, dtype=torch.complex64)
        gram = train_x.conj().T @ train_x + self.regularization * identity
        rhs = train_x.conj().T @ train_y
        weights = torch.linalg.solve(gram, rhs)
        return (windows @ weights).real.float()

