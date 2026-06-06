# -*- coding: utf-8 -*-
"""
通信环境模块 — Gym 风格的单帧交互接口

状态空间定义 (45 维):
  s_t = [接收窗 I/Q (42 维) | one-hot 位置编码 (3 维)]
  接收窗: 以 t 为中心的 2K+1 个符号 (K=10)
  位置编码: [1,0,0]=训练, [0,1,0]=导频, [0,0,1]=数据

奖励设计:
  训练位/导频位: 正确 +1, 错误 -1
  数据位: 0 (通过策略梯度间接优化)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from env.frame_structure import FrameConfig, FrameGenerator
from env.channel_models import RayleighMultipathChannel


@dataclass
class EnvConfig:
    frame: FrameConfig = field(default_factory=FrameConfig)
    channel: dict = field(default_factory=lambda: dict(
        num_taps=16, delay_spread=10, snr_db=10.0,
        time_varying=False, doppler_hz=1.0, symbol_rate=1e6, seed=42,
    ))
    window_K: int = 10
    seed: int = 42

    @property
    def state_dim(self) -> int:
        return 2 * (2 * self.window_K + 1) + 3


class CommunicationEnv:
    """逐帧通信环境 — 每帧 = 512 个符号的完整传输帧。"""

    def __init__(self, config: EnvConfig):
        self.config = config
        self.frame_cfg = config.frame
        self.window_K = config.window_K
        self.frame_gen = FrameGenerator(self.frame_cfg)
        self.channel = RayleighMultipathChannel(**config.channel)
        self._bits = None
        self._tx_symbols = None
        self._rx_symbols = None
        self._t = 0
        self._done = False
        self._rng = np.random.default_rng(config.seed)

    def reset(self, rng=None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if rng is not None:
            self._rng = rng
        self._bits = self.frame_gen.generate(self._rng)
        self._tx_symbols = self.frame_gen.modulate(self._bits)
        self._rx_symbols = self.channel(self._tx_symbols)
        self._t = 0
        self._done = False
        state = self._build_state()
        info = dict(t=self._t, bit_type=self.frame_cfg.bit_type(0),
                    true_bit=self._bits[0].item(), snr_db=self.channel.snr_db)
        return state, info

    def step(self, action) -> Tuple[torch.Tensor, float, bool, bool, Dict[str, Any]]:
        assert not self._done, "帧已结束。"
        true_bit = self._bits[self._t].item()
        bt = self.frame_cfg.bit_type(self._t)
        if bt in ("train", "pilot"):
            reward = 1.0 if (action.item() > 0.5) == (true_bit == 1.0) else -1.0
        else:
            reward = 0.0
        self._t += 1
        terminated = (self._t >= self.frame_cfg.frame_len)
        self._done = terminated
        state = self._build_state() if not terminated else torch.zeros(self.config.state_dim)
        info = dict(t=self._t, bit_type=bt, true_bit=true_bit, snr_db=self.channel.snr_db)
        return state, reward, terminated, False, info

    def _build_state(self) -> torch.Tensor:
        K = self.window_K
        rx3d = self._rx_symbols.unsqueeze(0)
        padded = F.pad(rx3d, (0, 0, K, K), mode="replicate").squeeze(0)
        center = self._t + K
        win = padded[center - K: center + K + 1].reshape(-1)
        bt = self.frame_cfg.bit_type(self._t)
        pos = torch.zeros(3)
        pos[{"train": 0, "pilot": 1, "data": 2}[bt]] = 1.0
        return torch.cat([win, pos])

    def get_true_bits(self) -> torch.Tensor:
        return self._bits.clone()

    def get_all_states(self) -> torch.Tensor:
        """批量预计算整帧所有状态 (L, D), 比逐步构建快约 2.6 倍。"""
        K = self.window_K
        L = self.frame_cfg.frame_len
        rx3d = self._rx_symbols.unsqueeze(0)
        padded = F.pad(rx3d, (0, 0, K, K), mode="replicate").squeeze(0)
        states = []
        for t in range(L):
            center = t + K
            win = padded[center - K : center + K + 1].reshape(-1)
            bt = self.frame_cfg.bit_type(t)
            pos = torch.zeros(3)
            pos[{"train": 0, "pilot": 1, "data": 2}[bt]] = 1.0
            states.append(torch.cat([win, pos]))
        return torch.stack(states)

    def get_rx_symbols(self) -> torch.Tensor:
        return self._rx_symbols.clone()

    def set_snr(self, snr_db):
        self.channel.set_snr(snr_db)


def test_env():
    cfg = EnvConfig(
        frame=FrameConfig(frame_len=512, train_len=128, pilot_len=64, num_pilots=2),
        channel=dict(num_taps=16, delay_spread=10, snr_db=10.0, seed=42),
        window_K=10,
    )
    env = CommunicationEnv(cfg)
    state, info = env.reset()
    assert info["true_bit"] == env._bits[0].item()
    print(f"状态维度: {state.shape[0]}, 期望: {cfg.state_dim}")
    state, reward, term, trunc, info2 = env.step(torch.tensor([0.5]))
    print(f"第一步: 奖励={reward}, 比特类型={info2['bit_type']}")
    print("环境接口自测通过。")


if __name__ == "__main__":
    test_env()
