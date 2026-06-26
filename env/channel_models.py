# -*- coding: utf-8 -*-
"""
信道模型模块 — 16抽头瑞利衰落信道 + 3GPP标准信道模型

包含两类信道:
  1. RayleighMultipathChannel:
     经典的16抽头频率选择性瑞利衰落信道
     指数衰减 PDP, 时延扩展可配, 适合基线对比

  2. ThreeGPPChannel:
     3GPP TR 38.901 标准信道模型
     支持 EPA / EVA / ETU 三种 Profile

两者接口完全兼容: reset_taps / set_taps / set_snr
/ convolve / add_awgn / __call__ / summary

卷积运算使用 F.conv1d 向量化实现。
"""

import math
import torch
import torch.nn.functional as F


# ========== 1. 经典瑞利衰落信道 ==========

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
        # 指数衰减 PDP: 首抽头最大, 末抽头接近 0
        self._pdp = torch.exp(-torch.linspace(0, 3, num_taps))
        self._pdp = self._pdp / self._pdp.sum()

    def reset_taps(self):
        real = torch.randn(self.num_taps) * self._pdp.sqrt()
        imag = torch.randn(self.num_taps) * self._pdp.sqrt()
        self._taps = torch.complex(real, imag)

    def set_taps(self, taps):
        self._taps = taps.clone()

    def update_taps_if_needed(self, t):
        if not self.time_varying:
            return
        rho = math.exp(-2 * math.pi * self.doppler_hz / self.symbol_rate)
        noise_real = torch.randn(self.num_taps) * (1 - rho**2) ** 0.5
        noise_imag = torch.randn(self.num_taps) * (1 - rho**2) ** 0.5
        noise = torch.complex(noise_real, noise_imag)
        self._taps = rho * self._taps + noise * self._pdp.sqrt()

    def convolve(self, symbols):
        L = symbols.shape[0]
        ir = symbols[:, 0].view(1, 1, L)
        ii = symbols[:, 1].view(1, 1, L)
        tr = self._taps.real.flip(0).view(1, 1, -1)
        ti = self._taps.imag.flip(0).view(1, 1, -1)
        pad_len = max(self.delay_spread, self.num_taps - 1)
        ir = F.pad(ir, (pad_len, 0))
        ii = F.pad(ii, (pad_len, 0))
        yr = (F.conv1d(ir, tr) - F.conv1d(ii, ti)).squeeze()
        yi = (F.conv1d(ir, ti) + F.conv1d(ii, tr)).squeeze()
        return torch.stack([yr, yi], dim=-1)

    def add_awgn(self, signal):
        signal_power = (signal ** 2).mean()
        noise_power = signal_power / self.snr_linear
        noise_std = torch.sqrt(noise_power / 2)
        noise = torch.randn_like(signal) * noise_std
        return signal + noise

    def __call__(self, symbols):
        self.reset_taps()
        received = self.convolve(symbols)
        received = self.add_awgn(received)
        return received

    def set_snr(self, snr_db):
        self.snr_db = snr_db
        self.snr_linear = 10.0 ** (snr_db / 10.0)

    def summary(self):
        return f"瑞利信道 {self.num_taps}抽头 时延扩展={self.delay_spread} SNR={self.snr_db}dB"


# ========== 2. 3GPP 标准信道模型 ==========

# EPA (Extended Pedestrian A): 7 抽头, 时延扩展 ~0.4 us
EPA = [(0,0),(30,-1),(70,-2),(90,-3),(110,-8),(190,-17.2),(410,-20.8)]
# EVA (Extended Vehicular A): 9 抽头, 时延扩展 ~1.5 us
EVA = [(0,0),(30,-1.5),(150,-1.4),(310,-3.6),(370,-0.6),(710,-9.1),(1090,-7),(1730,-12),(2510,-16.9)]
# ETU (Extended Typical Urban): 9 抽头, 时延扩展 ~5 us
ETU = [(0,-1),(50,-1),(120,-1),(200,0),(230,0),(500,0),(1600,-3),(2300,-5),(5000,-7)]

PROF = {
    "epa": {"pd": EPA, "d": "EPA 7抽头 步行环境"},
    "eva": {"pd": EVA, "d": "EVA 9抽头 车载环境"},
    "etu": {"pd": ETU, "d": "ETU 9抽头 典型城市"},
}


class ThreeGPPChannel:
    """3GPP 标准无线信道模型 (准静态块衰落).

    支持 3 种 Profile: epa / eva / etu
    时延按 ns 归一化到符号周期 (1e6 symbol/s => 1 us = 1000 ns)
    非均匀时延取整对齐到整数网格。
    """

    def __init__(self, profile="eva", snr_db=10.0,
                 time_varying=False, doppler_hz=70.0, symbol_rate=1e6,
                 seed=42, **kw):
        self.pn = profile.lower()
        if self.pn not in PROF:
            raise ValueError(f"未知profile: {self.pn}, 可选: {list(PROF.keys())}")
        self.pr = PROF[self.pn]
        self.snr_db = snr_db
        self.sl = 10.0 ** (snr_db / 10.0)
        self.tv = time_varying
        self.dp = doppler_hz
        self.sr = symbol_rate
        self._taps = None
        pd = self.pr["pd"]
        self.dl = torch.tensor([d / 1000.0 for d, _ in pd])
        self.nt = len(pd)
        pl = [10.0 ** (p / 10.0) for _, p in pd]
        pt = torch.tensor(pl)
        self.pdp = pt / pt.sum()
        self.md = int(self.dl.max().ceil().item())

    def reset_taps(self):
        r = torch.randn(self.nt) * self.pdp.sqrt()
        i = torch.randn(self.nt) * self.pdp.sqrt()
        self._taps = torch.complex(r, i)

    def set_taps(self, taps):
        self._taps = taps.clone()

    def set_snr(self, snr_db):
        self.snr_db = snr_db
        self.sl = 10.0 ** (snr_db / 10.0)

    def update_taps_if_needed(self, t):
        if not self.tv or self._taps is None:
            return
        rh = math.exp(-2 * math.pi * self.dp / self.sr)
        nr = torch.randn(self.nt) * (1 - rh ** 2) ** 0.5
        ni = torch.randn(self.nt) * (1 - rh ** 2) ** 0.5
        self._taps = rh * self._taps + torch.complex(nr, ni) * self.pdp.sqrt()

    def convolve(self, symbols):
        L = symbols.shape[0]
        ir = symbols[:, 0].view(1, 1, L)
        ii = symbols[:, 1].view(1, 1, L)
        tr = torch.zeros(self.md + 1)
        ti = torch.zeros(self.md + 1)
        for k, d in enumerate(self.dl):
            p = int(round(d.item()))
            if p <= self.md:
                tr[p] += self._taps[k].real
                ti[p] += self._taps[k].imag
        nz = (tr.abs() + ti.abs()) > 1e-10
        el = max(int(nz.sum().item()), 1)
        tr = tr.flip(0)[-el:].view(1, 1, -1)
        ti = ti.flip(0)[-el:].view(1, 1, -1)
        ir = F.pad(ir, (el - 1, 0))
        ii = F.pad(ii, (el - 1, 0))
        yr = (F.conv1d(ir, tr) - F.conv1d(ii, ti)).squeeze()
        yi = (F.conv1d(ir, ti) + F.conv1d(ii, tr)).squeeze()
        return torch.stack([yr[:L], yi[:L]], -1)

    def add_awgn(self, signal):
        sp = (signal ** 2).mean()
        np = sp / self.sl
        ns = torch.sqrt(np / 2)
        return signal + torch.randn_like(signal) * ns

    def __call__(self, symbols):
        self.reset_taps()
        r = self.convolve(symbols)
        return self.add_awgn(r)

    def summary(self):
        return ("3GPP " + self.pn.upper() + " | " + self.pr["d"]
                + " | SNR=" + str(self.snr_db) + "dB | 时延=" + str(self.md) + "sym")


# ========== 测试 ==========

def test_channel():
    """保留旧接口 test_channel(), 测试所有信道模型"""
    print("=" * 55)
    print("  信道模型测试")
    print("=" * 55)

    # 1. 瑞利信道
    print("\n[瑞利信道]")
    rc = RayleighMultipathChannel(num_taps=16, delay_spread=10, snr_db=20)
    st = torch.tensor([[1.0, 0.0], [-1.0, 0.0]] * 50)
    r = rc(st)
    print(" ", rc.summary(), "| 输入:", st.shape, "-> 输出:", r.shape)
    assert r.shape == (100, 2)

    # 2. 3GPP 信道
    print("\n[3GPP信道]")
    for n in ["epa", "eva", "etu"]:
        c = ThreeGPPChannel(profile=n, snr_db=15)
        r = c(st)
        print(" ", c.summary(), "| 输出:", r.shape)
        assert r.shape == st.shape, f"形状不匹配: {r.shape}"

    # 3. 接口兼容验证
    print("\n[接口兼容验证]")
    for label, ch in [("瑞利", rc), ("3GPP", ThreeGPPChannel(profile="eva", snr_db=10))]:
        ch.reset_taps()
        o1 = ch.convolve(st)
        o2 = ch.add_awgn(o1)
        ch.set_snr(5.0)
        ch.update_taps_if_needed(0)
        taps = ch._taps.clone()
        ch.set_taps(taps)
        print(" ", label, "全接口兼容 | after set_snr(5): snr_db=", ch.snr_db)

    print("\n信道模块自测通过。")


if __name__ == "__main__":
    test_channel()
