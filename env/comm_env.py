# -*- coding: utf-8 -*-
"""通信环境模块"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any
from env.frame_structure import FrameConfig, FrameGenerator
from env.rayleigh_channel import RayleighMultipathChannel
from env.rician_channel import RicianMultipathChannel
from env._3gpp_channel import ThreeGPPChannel

def _make_channel(cfg):
    channel_type = cfg.get("type")
    payload = {k: v for k, v in cfg.items() if k != "type"}
    if "profile" in payload:
        return ThreeGPPChannel(**payload)
    if channel_type == "rician" or "k_factor_db" in payload:
        return RicianMultipathChannel(**payload)
    return RayleighMultipathChannel(**payload)

@dataclass
class EnvConfig:
    frame: FrameConfig = field(default_factory=FrameConfig)
    channel: dict = field(default_factory=lambda: dict(
        num_taps=16, delay_spread=10, snr_db=10.0,
        time_varying=False, doppler_hz=1.0, symbol_rate=1e6, seed=42,
    ))
    window_K: int = 10
    seed: int = 42
    use_mmse_features: bool = False
    @property
    def state_dim(self):
        return 2 * (2 * self.window_K + 1) + 3 + (2 if self.use_mmse_features else 0)

class CommunicationEnv:
    def __init__(self, config):
        self.config = config
        self.frame_cfg = config.frame
        self.window_K = config.window_K
        self.frame_gen = FrameGenerator(self.frame_cfg)
        self.channel = _make_channel(config.channel)
        self._bits = None; self._tx_symbols = None; self._rx_symbols = None
        self._mmse_features = None
        self._t = 0; self._done = False
        self._rng = np.random.default_rng(config.seed)

    def reset(self, rng=None):
        if rng is not None: self._rng = rng
        self._bits = self.frame_gen.generate(self._rng)
        self._tx_symbols = self.frame_gen.modulate(self._bits)
        self._rx_symbols = self.channel(self._tx_symbols)
        self._mmse_features = self._build_mmse_features() if self.config.use_mmse_features else None
        self._t = 0; self._done = False
        state = self._build_state()
        info = dict(t=0, bit_type=self.frame_cfg.bit_type(0),
                    true_bit=self._bits[0].item(), snr_db=self.channel.snr_db)
        return state, info

    def step(self, action):
        assert not self._done
        bt = self.frame_cfg.bit_type(self._t)
        true_bit = self._bits[self._t].item()
        if bt in ("train","pilot"):
            reward = 1.0 if (action.item()>0.5)==(true_bit==1.0) else -1.0
        else: reward = 0.0
        self._t += 1
        term = (self._t >= self.frame_cfg.frame_len)
        self._done = term
        state = self._build_state() if not term else torch.zeros(self.config.state_dim)
        info = dict(t=self._t, bit_type=bt, true_bit=true_bit, snr_db=self.channel.snr_db)
        return state, reward, term, False, info

    def _build_state(self):
        K, rx = self.window_K, self._rx_symbols
        p = F.pad(rx.unsqueeze(0), (0,0,K,K), mode="replicate").squeeze(0)
        c = self._t + K
        win = p[c-K:c+K+1].reshape(-1)
        meta = self._state_meta(self._t)
        return torch.cat([win, meta, self._extra_features(self._t)])

    def get_true_bits(self): return self._bits.clone()
    def get_all_states(self):
        K, L = self.window_K, self.frame_cfg.frame_len
        p = F.pad(self._rx_symbols.unsqueeze(0), (0,0,K,K), mode="replicate").squeeze(0)
        states = []
        for t in range(L):
            c = t + K; win = p[c-K:c+K+1].reshape(-1)
            states.append(torch.cat([win, self._state_meta(t), self._extra_features(t)]))
        return torch.stack(states)
    def get_rx_symbols(self): return self._rx_symbols.clone()
    def set_snr(self, s): self.channel.set_snr(s)

    def _state_meta(self, t):
        bt = self.frame_cfg.bit_type(t)
        if bt in ("train", "pilot"):
            known_symbol = 1.0 - 2.0 * float(self._bits[t].item())
            return torch.tensor([known_symbol, 1.0, 0.0], dtype=torch.float32)
        return torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    def _extra_features(self, t):
        if self._mmse_features is None:
            return torch.empty(0, dtype=torch.float32)
        return self._mmse_features[t]

    def _build_mmse_features(self):
        from baseline.mmse_equalizer import MMSEEqualizer

        num_taps = getattr(self.channel, "num_taps", getattr(self.channel, "nt", 16))
        mmse = MMSEEqualizer(num_taps=num_taps)
        soft, _ = mmse(
            self._rx_symbols,
            self._bits[:self.frame_cfg.train_len],
            self._rx_symbols[:self.frame_cfg.train_len],
            self.channel.snr_db,
        )
        scale = soft.abs().detach().quantile(0.95).clamp_min(1e-3)
        soft_norm = torch.clamp(soft / scale, -3.0, 3.0) / 3.0
        conf = torch.sigmoid(soft.abs())
        return torch.stack([soft_norm.float(), conf.float()], dim=-1)

def test_env():
    print("瑞利信道:"); e=CommunicationEnv(EnvConfig()); s,_=e.reset()
    print(" ",e.channel.summary(),"state:",s.shape[0])
    print("3GPP ETU:"); e2=CommunicationEnv(EnvConfig(channel=dict(profile="etu",snr_db=10,seed=42))); s2,_=e2.reset()
    print(" ",e2.channel.summary(),"state:",s2.shape[0])
    print("3GPP EPA:"); e3=CommunicationEnv(EnvConfig(channel=dict(profile="epa",snr_db=15,seed=42))); s3,_=e3.reset()
    print(" ",e3.channel.summary(),"state:",s3.shape[0])
    print("环境接口自测通过。")

if __name__=="__main__":test_env()
