# -*- coding: utf-8 -*-
"""常用传统均衡算法 baseline。

这些算法只用于仿真对比和可选教师信号，不参与在线 RL reward。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from baseline.mmse_equalizer import MMSEEqualizer


def _bits_to_bpsk(bits: torch.Tensor) -> torch.Tensor:
    return (1.0 - 2.0 * bits.float()).to(torch.complex64)


def _rx_complex(rx: torch.Tensor) -> torch.Tensor:
    return torch.complex(rx[:, 0].float(), rx[:, 1].float())


def _estimate_channel(tx_sym: torch.Tensor, rx: torch.Tensor, num_taps: int) -> torch.Tensor:
    rxc = _rx_complex(rx)
    h = torch.zeros(num_taps, dtype=torch.complex64)
    denom = tx_sym.abs().square().sum().clamp_min(1e-6)
    for tap in range(num_taps):
        if tap < len(tx_sym):
            h[tap] = (rxc[tap:] * tx_sym[: len(tx_sym) - tap].conj()).sum() / denom
    return h


def _freq_equalize(rx: torch.Tensor, h: torch.Tensor, mode: str, snr_db: float = 10.0) -> torch.Tensor:
    rxc = _rx_complex(rx)
    n_fft = int(2 ** torch.ceil(torch.log2(torch.tensor(max(len(rxc), len(h), 8), dtype=torch.float32))).item())
    h_pad = torch.zeros(n_fft, dtype=torch.complex64)
    h_pad[: len(h)] = h
    H = torch.fft.fft(h_pad)
    power = H.abs().square()
    eps = 1e-4
    if mode == "matched_filter":
        W = H.conj()
    elif mode == "zero_forcing":
        W = H.conj() / (power + eps)
    elif mode == "mmse_like":
        snr_lin = 10 ** (snr_db / 10.0)
        W = H.conj() / (power + 1.0 / snr_lin + eps)
    else:
        raise ValueError(f"未知频域均衡模式: {mode}")
    soft = torch.fft.ifft(W * torch.fft.fft(rxc, n=n_fft)).real[: len(rxc)]
    return soft.float()


def _apply_fir(rx: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    rxc = _rx_complex(rx)
    filt_len = len(weights)
    half = filt_len // 2
    padded = torch.nn.functional.pad(rxc, (half, half), mode="constant", value=0)
    out = []
    for idx in range(len(rxc)):
        window = padded[idx : idx + filt_len]
        out.append((weights.conj() * window).sum().real)
    return torch.stack(out).float()


@dataclass
class FrequencyEqualizer:
    name: str
    mode: str
    num_taps: int = 16

    def __call__(self, rx: torch.Tensor, tx_tr_bits: torch.Tensor, rx_tr: torch.Tensor, snr_db: float):
        tx_sym = _bits_to_bpsk(tx_tr_bits)
        h = _estimate_channel(tx_sym, rx_tr, self.num_taps)
        return _freq_equalize(rx, h, self.mode, snr_db), h


@dataclass
class LMSFIREqualizer:
    num_taps: int = 16
    lr: float = 0.08
    epochs: int = 3

    def __call__(self, rx: torch.Tensor, tx_tr_bits: torch.Tensor, rx_tr: torch.Tensor, snr_db: float):
        filt_len = self.num_taps if self.num_taps % 2 == 1 else self.num_taps + 1
        weights = torch.zeros(filt_len, dtype=torch.complex64)
        weights[filt_len // 2] = 1.0 + 0.0j
        target = _bits_to_bpsk(tx_tr_bits)
        rxc = _rx_complex(rx_tr)
        half = filt_len // 2
        padded = torch.nn.functional.pad(rxc, (half, half), mode="constant", value=0)
        for _ in range(self.epochs):
            for idx in range(len(target)):
                window = padded[idx : idx + filt_len]
                pred = (weights.conj() * window).sum()
                err = target[idx] - pred
                norm = window.abs().square().sum().real + 1e-4
                weights = weights + self.lr * window * err.conj() / norm
        return _apply_fir(rx, weights), weights


@dataclass
class RLSFIREqualizer:
    num_taps: int = 16
    forgetting: float = 0.99
    delta: float = 10.0

    def __call__(self, rx: torch.Tensor, tx_tr_bits: torch.Tensor, rx_tr: torch.Tensor, snr_db: float):
        filt_len = self.num_taps if self.num_taps % 2 == 1 else self.num_taps + 1
        weights = torch.zeros(filt_len, dtype=torch.complex64)
        weights[filt_len // 2] = 1.0 + 0.0j
        P = torch.eye(filt_len, dtype=torch.complex64) * self.delta
        target = _bits_to_bpsk(tx_tr_bits)
        rxc = _rx_complex(rx_tr)
        half = filt_len // 2
        padded = torch.nn.functional.pad(rxc, (half, half), mode="constant", value=0)
        for idx in range(len(target)):
            x = padded[idx : idx + filt_len].unsqueeze(1)
            denom = self.forgetting + (x.conj().T @ P @ x).squeeze()
            gain = (P @ x / denom).squeeze(1)
            pred = (weights.conj() * x.squeeze(1)).sum()
            err = target[idx] - pred
            weights = weights + gain * err.conj()
            P = (P - gain.unsqueeze(1) @ x.conj().T @ P) / self.forgetting
        return _apply_fir(rx, weights), weights


def make_traditional_equalizers(num_taps: int = 16) -> Dict[str, object]:
    """构造用于论文对比的传统均衡算法集合。"""
    return {
        "matched_filter": FrequencyEqualizer("matched_filter", "matched_filter", num_taps=num_taps),
        "zero_forcing": FrequencyEqualizer("zero_forcing", "zero_forcing", num_taps=num_taps),
        "lms": LMSFIREqualizer(num_taps=num_taps),
        "rls": RLSFIREqualizer(num_taps=num_taps),
        "mmse": MMSEEqualizer(num_taps=num_taps),
    }
