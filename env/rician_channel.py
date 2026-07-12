# -*- coding: utf-8 -*-
"""莱斯多径信道模型。

该信道用于离线预训练的泛化训练：主径包含确定性 LOS 分量，其余散射分量按
Rayleigh 多径抽样。接口与 RayleighMultipathChannel 保持一致，便于环境层统一调用。
"""

import math

import torch
import torch.nn.functional as F


class RicianMultipathChannel:
    """频率选择性莱斯衰落信道。"""

    def __init__(
        self,
        num_taps=12,
        delay_spread=8,
        k_factor_db=6.0,
        snr_db=10.0,
        time_varying=False,
        doppler_hz=1.0,
        symbol_rate=1e6,
        seed=42,
        type=None,
    ):
        self.num_taps = int(num_taps)
        self.delay_spread = int(delay_spread)
        self.k_factor_db = float(k_factor_db)
        self.k_factor = 10.0 ** (self.k_factor_db / 10.0)
        self.snr_db = float(snr_db)
        self.snr_linear = 10.0 ** (self.snr_db / 10.0)
        self.time_varying = bool(time_varying)
        self.doppler_hz = float(doppler_hz)
        self.symbol_rate = float(symbol_rate)
        self.seed = seed
        self._taps = None

        self._pdp = torch.exp(-torch.linspace(0, 3, self.num_taps))
        self._pdp = self._pdp / self._pdp.sum()
        self._los = torch.zeros(self.num_taps, dtype=torch.complex64)
        self._los[0] = torch.sqrt(self._pdp[0]).to(torch.complex64)

    def reset_taps(self):
        scatter_r = torch.randn(self.num_taps) * self._pdp.sqrt()
        scatter_i = torch.randn(self.num_taps) * self._pdp.sqrt()
        scatter = torch.complex(scatter_r, scatter_i)
        los_scale = math.sqrt(self.k_factor / (self.k_factor + 1.0))
        scatter_scale = math.sqrt(1.0 / (self.k_factor + 1.0))
        self._taps = los_scale * self._los + scatter_scale * scatter

    def set_taps(self, taps):
        self._taps = taps.clone()

    def update_taps_if_needed(self, t):
        if not self.time_varying or self._taps is None:
            return
        rho = math.exp(-2 * math.pi * self.doppler_hz / self.symbol_rate)
        scatter_r = torch.randn(self.num_taps) * (1 - rho**2) ** 0.5
        scatter_i = torch.randn(self.num_taps) * (1 - rho**2) ** 0.5
        scatter = torch.complex(scatter_r, scatter_i) * self._pdp.sqrt()
        los_scale = math.sqrt(self.k_factor / (self.k_factor + 1.0))
        scatter_scale = math.sqrt(1.0 / (self.k_factor + 1.0))
        target = los_scale * self._los + scatter_scale * scatter
        self._taps = rho * self._taps + (1.0 - rho) * target

    def convolve(self, symbols):
        if self._taps is None:
            self.reset_taps()
        length = symbols.shape[0]
        in_r = symbols[:, 0].view(1, 1, length)
        in_i = symbols[:, 1].view(1, 1, length)
        tap_r = self._taps.real.flip(0).view(1, 1, -1)
        tap_i = self._taps.imag.flip(0).view(1, 1, -1)
        pad = max(self.delay_spread, self.num_taps - 1)
        in_r = F.pad(in_r, (pad, 0))
        in_i = F.pad(in_i, (pad, 0))
        out_r = (F.conv1d(in_r, tap_r) - F.conv1d(in_i, tap_i)).squeeze()
        out_i = (F.conv1d(in_r, tap_i) + F.conv1d(in_i, tap_r)).squeeze()
        return torch.stack([out_r[:length], out_i[:length]], -1)

    def add_awgn(self, signal):
        signal_power = (signal ** 2).mean()
        noise_power = signal_power / self.snr_linear
        return signal + torch.randn_like(signal) * torch.sqrt(noise_power / 2)

    def __call__(self, symbols):
        self.reset_taps()
        return self.add_awgn(self.convolve(symbols))

    def set_snr(self, snr_db):
        self.snr_db = float(snr_db)
        self.snr_linear = 10.0 ** (self.snr_db / 10.0)

    def summary(self):
        return (
            f"莱斯信道 {self.num_taps}抽头 K={self.k_factor_db:.1f}dB "
            f"时延扩展={self.delay_spread} SNR={self.snr_db}dB"
        )


def test():
    torch.manual_seed(42)
    channel = RicianMultipathChannel(num_taps=10, delay_spread=7, k_factor_db=6, snr_db=20)
    symbols = torch.tensor([[1.0, 0.0], [-1.0, 0.0]] * 50)
    received = channel(symbols)
    assert received.shape == (100, 2)
    channel.set_snr(5)
    print(channel.summary(), "| 输入:", symbols.shape, "-> 输出:", received.shape)


if __name__ == "__main__":
    test()
