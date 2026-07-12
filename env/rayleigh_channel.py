# -*- coding: utf-8 -*-
"""
经典瑞利衰落信道模型

16 抽头频率选择性瑞利衰落信道
指数衰减 PDP, 时延扩展可配
卷积运算使用 F.conv1d 向量化实现
"""
import math
import torch
import torch.nn.functional as F


class RayleighMultipathChannel:
    """频率选择性瑞利衰落信道。每帧重新抽取信道抽头 (准静态块衰落)。"""

    def __init__(self, num_taps=16, delay_spread=10, snr_db=10.0,
                 time_varying=False, doppler_hz=1.0, symbol_rate=1e6, seed=42):
        self.num_taps = num_taps
        self.delay_spread = delay_spread
        self.snr_db = snr_db
        self.snr_linear = 10.0 ** (snr_db / 10.0)
        self.time_varying = time_varying
        self.doppler_hz = doppler_hz
        self.symbol_rate = symbol_rate
        self._taps = None
        # 指数衰减 PDP
        self._pdp = torch.exp(-torch.linspace(0, 3, num_taps))
        self._pdp = self._pdp / self._pdp.sum()

    def reset_taps(self):
        r = torch.randn(self.num_taps) * self._pdp.sqrt()
        i = torch.randn(self.num_taps) * self._pdp.sqrt()
        self._taps = torch.complex(r, i)

    def set_taps(self, taps):
        self._taps = taps.clone()

    def update_taps_if_needed(self, t):
        if not self.time_varying: return
        rho = math.exp(-2 * math.pi * self.doppler_hz / self.symbol_rate)
        nr = torch.randn(self.num_taps) * (1 - rho**2) ** 0.5
        ni = torch.randn(self.num_taps) * (1 - rho**2) ** 0.5
        self._taps = rho * self._taps + torch.complex(nr, ni) * self._pdp.sqrt()

    def convolve(self, symbols):
        L = symbols.shape[0]
        ir = symbols[:, 0].view(1, 1, L)
        ii = symbols[:, 1].view(1, 1, L)
        tr = self._taps.real.flip(0).view(1, 1, -1)
        ti = self._taps.imag.flip(0).view(1, 1, -1)
        pad = max(self.delay_spread, self.num_taps - 1)
        ir = F.pad(ir, (pad, 0)); ii = F.pad(ii, (pad, 0))
        yr = (F.conv1d(ir, tr) - F.conv1d(ii, ti)).squeeze()
        yi = (F.conv1d(ir, ti) + F.conv1d(ii, tr)).squeeze()
        return torch.stack([yr[:L], yi[:L]], -1)

    def add_awgn(self, signal):
        sp = (signal ** 2).mean()
        np = sp / self.snr_linear
        return signal + torch.randn_like(signal) * torch.sqrt(np / 2)

    def __call__(self, symbols):
        self.reset_taps()
        r = self.convolve(symbols)
        return self.add_awgn(r)

    def set_snr(self, snr_db):
        self.snr_db = snr_db
        self.snr_linear = 10.0 ** (snr_db / 10.0)

    def summary(self):
        return f"瑞利信道 {self.num_taps}抽头 时延扩展={self.delay_spread} SNR={self.snr_db}dB"


def test():
    torch.manual_seed(42)
    c = RayleighMultipathChannel(num_taps=16, delay_spread=10, snr_db=20)
    s = torch.tensor([[1.,0.],[-1.,0.]]*50)
    r = c(s)
    print(c.summary(), "| 输入:", s.shape, "-> 输出:", r.shape)
    assert r.shape == (100, 2)
    c.set_snr(5); c.reset_taps(); c.convolve(s); c.add_awgn(s)
    print("瑞利信道自测通过。")

if __name__ == "__main__": test()
