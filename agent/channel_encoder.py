# -*- coding: utf-8 -*-
"""
信道编码器 — 从接收到的训练序列中提取信道特征

输入: 接收到的训练序列 (I/Q, 128x2) + 已知训练比特
输出: cond_dim 维的信道特征向量

用途: 在RL状态中加入该特征, 辅助网络做信道自适应均衡
"""
import torch
import torch.nn as nn


class ChannelEncoder(nn.Module):
    """基于训练序列的信道特征提取器.

    结构:
      [接收I/Q (128x2), 已知训练残差(128)] -> Conv1x3 + Pool -> FC -> cond_dim
    """

    def __init__(self, train_len=128, cond_dim=8):
        super().__init__()
        self.train_len = train_len

        # 时域特征提取
        self.conv = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(64),
        )
        # 全连接映射到信道条件向量
        self.fc = nn.Sequential(
            nn.Linear(16 * 64 + train_len, 64),
            nn.ReLU(),
            nn.Linear(64, cond_dim),
            nn.Tanh(),
        )

    def forward(self, rx_train, known_bits):
        """
        rx_train: (B, train_len, 2) — 接收训练符号 I/Q
        known_bits: (B, train_len) — 已知训练比特 (±1)
        return: (B, cond_dim)
        """
        B = rx_train.shape[0]
        # I/Q 时域特征
        x = rx_train.permute(0, 2, 1)  # (B, 2, train_len)
        feat = self.conv(x).view(B, -1)  # (B, 16*64)
        # 已知训练比特
        kb = known_bits.view(B, -1)
        combined = torch.cat([feat, kb], dim=1)
        cond = self.fc(combined)
        return cond


def test():
    enc = ChannelEncoder(train_len=128, cond_dim=8)
    rx = torch.randn(2, 128, 2)
    kb = torch.randint(0, 2, (2, 128)).float() * 2 - 1
    cond = enc(rx, kb)
    print(f"信道编码器: rx{rx.shape} + kb{kb.shape} -> cond{cond.shape}")
    print("参数:", sum(p.numel() for p in enc.parameters()))
    print("测试通过")

if __name__ == "__main__": test()
