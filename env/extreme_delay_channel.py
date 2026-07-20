# -*- coding: utf-8 -*-
"""EME 启发的稀疏长时延扩展信道。"""

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class ExtremeDelayChannelConfig:
    """稀疏长回波信道参数。"""

    max_delay_symbols: int = 40
    min_paths: int = 3
    max_paths: int = 7
    snr_db: float = 10.0
    gauss_markov_rho: float = 0.995
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_delay_symbols < 1:
            raise ValueError("最大相对时延必须至少为 1 个符号。")
        if self.min_paths < 2 or self.max_paths < self.min_paths:
            raise ValueError("路径数范围无效。")
        if self.max_paths > self.max_delay_symbols + 1:
            raise ValueError("路径数不能超过可用的离散时延位置数。")
        if not 0.0 <= self.gauss_markov_rho <= 1.0:
            raise ValueError("Gauss-Markov 相关系数必须位于 [0, 1]。")


class ExtremeDelayChannel:
    """具有跨帧发送历史和慢变路径增益的因果复数信道。"""

    def __init__(self, config: ExtremeDelayChannelConfig):
        self.config = config
        self.snr_db = float(config.snr_db)
        self._rng = np.random.default_rng(config.seed)
        self.delays: list[int] = []
        self.taps = torch.empty(0, dtype=torch.complex64)
        self._history = torch.zeros(config.max_delay_symbols, dtype=torch.complex64)
        self._frame_index = 0

    def reset_episode(self, warmup_symbols: torch.Tensor | None = None) -> None:
        path_count = int(self._rng.integers(self.config.min_paths, self.config.max_paths + 1))
        interior_count = path_count - 2
        if interior_count:
            interior = self._rng.choice(
                np.arange(1, self.config.max_delay_symbols),
                size=interior_count,
                replace=False,
            ).tolist()
        else:
            interior = []
        self.delays = sorted([0, self.config.max_delay_symbols, *map(int, interior)])

        delays_np = np.asarray(self.delays, dtype=np.float64)
        decay = np.exp(-delays_np / max(self.config.max_delay_symbols * 0.65, 1.0))
        shadow = np.exp(self._rng.normal(0.0, 0.65, size=path_count))
        powers = decay * shadow
        powers = powers / powers.sum()
        phases = self._rng.uniform(-math.pi, math.pi, size=path_count)
        taps_np = np.sqrt(powers) * np.exp(1j * phases)
        self.taps = torch.from_numpy(taps_np.astype(np.complex64))
        self.taps = self.taps / self.taps.abs().square().sum().sqrt().clamp_min(1e-8)
        self._frame_index = 0
        if warmup_symbols is None:
            self._history = torch.zeros(
                self.config.max_delay_symbols, dtype=torch.complex64
            )
        else:
            warmup = self._to_complex(warmup_symbols)
            if warmup.numel() < self.config.max_delay_symbols:
                raise ValueError("warm-up 符号数量不足以填满信道历史。")
            self._history = warmup[-self.config.max_delay_symbols :].clone()

    def set_impulse_response(self, delays: list[int], taps: torch.Tensor) -> None:
        if len(delays) != int(taps.numel()):
            raise ValueError("时延数量与抽头数量不一致。")
        if min(delays) < 0 or max(delays) > self.config.max_delay_symbols:
            raise ValueError("指定时延超出信道范围。")
        self.delays = [int(value) for value in delays]
        self.taps = taps.to(torch.complex64).clone()

    def set_snr(self, snr_db: float) -> None:
        self.snr_db = float(snr_db)

    def _evolve_taps(self) -> None:
        rho = float(self.config.gauss_markov_rho)
        if rho >= 1.0 or self.taps.numel() == 0:
            return
        noise = self._rng.normal(size=self.taps.numel()) + 1j * self._rng.normal(
            size=self.taps.numel()
        )
        noise_t = torch.from_numpy(noise.astype(np.complex64))
        powers = self.taps.abs().square().clamp_min(1e-8)
        candidate = rho * self.taps + math.sqrt(1.0 - rho * rho) * noise_t * powers.sqrt()
        self.taps = candidate / candidate.abs().square().sum().sqrt().clamp_min(1e-8)

    @staticmethod
    def _to_complex(symbols: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(symbols):
            return symbols.to(torch.complex64)
        if symbols.ndim != 2 or symbols.shape[-1] != 2:
            raise ValueError("符号必须为复数向量或形状 (T, 2) 的 I/Q 张量。")
        return torch.complex(symbols[:, 0].float(), symbols[:, 1].float())

    def transmit(self, symbols: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        if self.taps.numel() == 0:
            self.reset_episode()
        if self._frame_index > 0:
            self._evolve_taps()
        current = self._to_complex(symbols)
        memory = self.config.max_delay_symbols
        extended = torch.cat([self._history.to(current.device), current])
        output = torch.zeros_like(current)
        for delay, tap in zip(self.delays, self.taps.to(current.device)):
            start = memory - int(delay)
            output = output + tap * extended[start : start + current.numel()]
        self._history = extended[-memory:].detach().cpu().clone()
        self._frame_index += 1

        if add_noise:
            signal_power = output.abs().square().mean().clamp_min(1e-12)
            noise_power = signal_power / (10.0 ** (self.snr_db / 10.0))
            noise_np = self._rng.normal(size=current.numel()) + 1j * self._rng.normal(
                size=current.numel()
            )
            noise = torch.from_numpy(noise_np.astype(np.complex64)).to(current.device)
            output = output + noise * torch.sqrt(noise_power / 2.0)
        return torch.stack([output.real.float(), output.imag.float()], dim=-1)

    __call__ = transmit

    def summary(self) -> str:
        return (
            f"极端稀疏长回波信道: paths={len(self.delays)}, "
            f"max_delay={self.config.max_delay_symbols}, SNR={self.snr_db:.1f} dB"
        )

