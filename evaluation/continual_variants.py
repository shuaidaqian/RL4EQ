# -*- coding: utf-8 -*-
"""匹配物理状态下拆分均衡器神经分支，并校验候选比较清单。"""

from __future__ import annotations

import math

import torch
from torch import nn

from training.continual_receiver import ARMS


def validate_variants(manifest):
    variants = manifest.get('variants', [])
    names = [row.get('name') for row in variants]
    if not names or len(set(names)) != len(names) or any(not isinstance(x, str) or not x for x in names):
        raise ValueError('候选名必须非空、互异。')
    previous = {}
    for row in variants:
        if row.get('arm') not in ARMS or not row.get('checkpoint'):
            raise ValueError('每个候选必须声明接收臂与 checkpoint。')
        if row.get('decoder', 'neural') not in {'neural', 'physics_branch'}:
            raise ValueError('未知解码分支。')
        if row.get('phase_estimator', 'unwrap_median') not in {'unwrap_median', 'circular_all', 'circular_clean'}:
            raise ValueError('未知相位估计器。')
        if row.get('operator_backend', 'direct') not in {'direct', 'fft'}:
            raise ValueError('线性算子后端须为 direct 或 fft。')
        noise = row.get('process_noise', 1e-4)
        if not isinstance(noise, (float, int)) or not math.isfinite(noise) or noise < 0:
            raise ValueError('过程噪声须为非负有限数。')
        source = row.get('context_source')
        if source is not None:
            if source not in previous or row.get('decoder') != 'physics_branch':
                raise ValueError('物理分支共享上下文必须引用排在前面的候选。')
            if row['checkpoint'] != previous[source]['checkpoint']:
                raise ValueError('共享上下文的分支消融必须使用同一 checkpoint。')
        previous[row['name']] = row
    for pair in manifest.get('pairs', []):
        if len(pair) != 2 or pair[0] not in names or pair[1] not in names or pair[0] == pair[1]:
            raise ValueError('配对差必须引用两个不同且已声明的候选。')
    return variants


class PhysicsBranch(nn.Module):
    """仅保留相同同步分支和物理 warm start，移除展开 Transformer 修正。

    同步仍含 checkpoint 的已学习相位修正，不能称完全非神经接收机。
    本消融回答 Transformer 修正的增量，不覆盖所有学习模块的贡献。
    """

    def __init__(self, model):
        super().__init__()
        self.model = model
        if model.config.enable_channel_state_branch or model.config.enable_symbol_residual_branch:
            raise ValueError('当前物理分支消融不支持额外学习状态或符号残差模块。')

    @property
    def config(self):
        return self.model.config

    def forward(self, rx_iq, condition, region_ids, soft_tail, *, adapt_symbols, adapt_mask):
        corrected = self.model._apply_phase_correction(rx_iq, condition)
        if self.model.online_phase_trend_adapter is not None:
            corrected = self.model.online_phase_trend_adapter(corrected)
        rx = torch.complex(corrected[..., 0], corrected[..., 1])
        logits = self.model._physics_warm_start_logits(rx, condition, soft_tail)
        return logits, logits.sigmoid()


class ForwardCounter(nn.Module):
    """对实际模型调用计数，避免用写死诊断值证明一次前向。"""

    def __init__(self, model):
        super().__init__()
        self.model, self.calls = model, 0

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        self.calls += 1
        return self.model(*args, **kwargs)
