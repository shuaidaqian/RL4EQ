# -*- coding: utf-8 -*-
"""
3GPP 标准无线信道模型

基于 3GPP TR 38.901 / 36.101 标准:
  - EPA (Extended Pedestrian A): 7抽头, 步行环境
  - EVA (Extended Vehicular A): 9抽头, 车载环境
  - ETU (Extended Typical Urban): 9抽头, 典型城市

时延按 ns 归一化到符号周期 (1e6 symbol/s => 1 us = 1000 ns)
非均匀时延取整对齐到整数网格
卷积运算使用 F.conv1d 向量化实现
"""
import math
import torch
import torch.nn.functional as F


# 3GPP 标准功率延迟分布 (时延 ns, 功率 dB)
EPA_PROFILE = [(0,0),(30,-1),(70,-2),(90,-3),(110,-8),(190,-17.2),(410,-20.8)]
EVA_PROFILE = [(0,0),(30,-1.5),(150,-1.4),(310,-3.6),(370,-0.6),(710,-9.1),(1090,-7),(1730,-12),(2510,-16.9)]
ETU_PROFILE = [(0,-1),(50,-1),(120,-1),(200,0),(230,0),(500,0),(1600,-3),(2300,-5),(5000,-7)]

PROFILES = {
    "epa": {"pd": EPA_PROFILE, "desc": "EPA 7抽头 步行环境"},
    "eva": {"pd": EVA_PROFILE, "desc": "EVA 9抽头 车载环境"},
    "etu": {"pd": ETU_PROFILE, "desc": "ETU 9抽头 典型城市"},
}


class ThreeGPPChannel:
    """3GPP 标准无线信道模型 (准静态块衰落)。

    参数:
        profile: str — "epa" / "eva" / "etu"
        snr_db: float — 信噪比 (dB)
        time_varying: bool — 是否启用 Jakes 时变
        doppler_hz: float — 多普勒频移
        symbol_rate: float — 符号速率
    """

    def __init__(self, profile="eva", snr_db=10.0,
                 time_varying=False, doppler_hz=70.0, symbol_rate=1e6, seed=42):
        pn = profile.lower()
        if pn not in PROFILES:
            raise ValueError(f"未知profile: {pn}, 可选: {list(PROFILES.keys())}")
        self.pn = pn
        self.pr = PROFILES[pn]
        self.snr_db = snr_db
        self.sl = 10.0 ** (snr_db / 10.0)
        self.tv = time_varying
        self.dp = doppler_hz
        self.sr = symbol_rate
        self._taps = None
        pd = self.pr["pd"]
        self.dl = torch.tensor([d / 1000.0 for d, _ in pd])
        self.nt = len(pd)
        pl = torch.tensor([10.0 ** (p / 10.0) for _, p in pd])
        self.pdp = pl / pl.sum()
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
        if not self.tv or self._taps is None: return
        rh = math.exp(-2 * math.pi * self.dp / self.sr)
        nr = torch.randn(self.nt) * (1 - rh**2) ** 0.5
        ni = torch.randn(self.nt) * (1 - rh**2) ** 0.5
        self._taps = rh * self._taps + torch.complex(nr, ni) * self.pdp.sqrt()

    def convolve(self, symbols):
        """多径卷积，支持非均匀时延 (取整对齐到整数网格)。"""
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
        return signal + torch.randn_like(signal) * torch.sqrt(sp / self.sl / 2)

    def __call__(self, symbols):
        self.reset_taps()
        r = self.convolve(symbols)
        return self.add_awgn(r)

    def summary(self):
        return (f"3GPP {self.pn.upper()} | {self.pr["desc"]}"
                f" | SNR={self.snr_db}dB | 时延={self.md}sym")


def test():
    torch.manual_seed(42)
    s = torch.tensor([[1.,0.],[-1.,0.]]*50)
    for n in ["epa","eva","etu"]:
        c = ThreeGPPChannel(profile=n, snr_db=15)
        r = c(s)
        print(c.summary(), "| 输出:", r.shape)
        assert r.shape == s.shape
    print("3GPP信道自测通过。")

if __name__ == "__main__": test()
