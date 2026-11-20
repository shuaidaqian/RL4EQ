# -*- coding: utf-8 -*-
"""匹配物理状态下拆分均衡器神经分支，并校验候选比较清单。"""

from __future__ import annotations

import copy
import math
from dataclasses import replace

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
        current_prior = row.get('current_pilot_prior', 'acquisition')
        if current_prior not in {'acquisition', 'aged_acquisition'}:
            raise ValueError('未知当前 Pilot 先验。')
        if row['arm'] != 'current_pilot' and current_prior != 'acquisition':
            raise ValueError('老化 acquisition 先验只适用于 current_pilot 对照。')
        if row.get('decoder', 'neural') not in {
            'neural', 'physics_branch', 'known_adapt_real_physics', 'known_adapt_real_neural',
        }:
            raise ValueError('未知解码分支。')
        solver_iterations = row.get('physics_solver_iterations')
        solver_diagnostics = row.get('physics_solver_diagnostics', False)
        if solver_iterations is not None and (type(solver_iterations) is not int or solver_iterations <= 0):
            raise ValueError('物理解码迭代次数须为正整数。')
        if type(solver_diagnostics) is not bool:
            raise ValueError('物理求解诊断开关须为布尔类型。')
        if (solver_iterations is not None or solver_diagnostics) and row.get('decoder') != 'known_adapt_real_physics':
            raise ValueError('显式物理求解对照只适用于 known_adapt_real_physics。')
        if row.get('phase_estimator', 'unwrap_median') not in {'unwrap_median', 'circular_all', 'circular_clean'}:
            raise ValueError('未知相位估计器。')
        if row.get('operator_backend', 'direct') not in {'direct', 'fft'}:
            raise ValueError('线性算子后端须为 direct 或 fft。')
        if row.get('phase_reference_backend', 'vectorized') not in {'legacy', 'vectorized'}:
            raise ValueError('相位参考实现须为 legacy 或 vectorized。')
        noise = row.get('process_noise', 1e-4)
        if not isinstance(noise, (float, int)) or not math.isfinite(noise) or noise < 0:
            raise ValueError('过程噪声须为非负有限数。')
        initial_intervals = row.get('initial_prediction_intervals', 1.0)
        if (not isinstance(initial_intervals, (float, int))
                or not math.isfinite(initial_intervals) or initial_intervals < 0):
            raise ValueError('首帧预测的转移间隔须为非负有限数。')
        state_model = row.get('state_model', 'random_walk')
        if state_model not in {'random_walk', 'ar1'}:
            raise ValueError('CIR 状态预测模型须为 random_walk 或 ar1。')
        if state_model == 'ar1':
            rho = row.get('state_rho')
            power = row.get('stationary_power_by_tap')
            if not isinstance(rho, (float, int)) or not math.isfinite(rho) or not 0 <= rho < 1:
                raise ValueError('AR(1) 候选必须给出 [0,1) 内的有限 state_rho。')
            if not isinstance(power, list) or not power or any(
                not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0
                for value in power
            ):
                raise ValueError('AR(1) 候选必须给出非负有限 stationary_power_by_tap。')
        source = row.get('context_source')
        if source is not None:
            if source not in previous:
                raise ValueError('共享上下文必须引用排在前面的候选。')
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


class KnownAdaptRealPhysicsBranch(PhysicsBranch):
    """固定合法 Adapt 符号，并在实值 BPSK 子空间求物理 warm-start。"""

    def __init__(self, model, *, solver_iterations=None, solver_diagnostics=False):
        if solver_iterations is not None:
            if type(solver_iterations) is not int or solver_iterations <= 0:
                raise ValueError('物理解码迭代次数须为正整数。')
            # 候选可能共享缓存的checkpoint；配置覆盖必须局限于本分支。
            model = copy.deepcopy(model)
            model.config = replace(model.config, physics_warm_start_iterations=solver_iterations)
        super().__init__(model)
        if type(solver_diagnostics) is not bool:
            raise ValueError('物理求解诊断开关须为布尔类型。')
        if solver_diagnostics and model.config.physics_warm_start_iterations <= 0:
            raise ValueError('求解诊断要求正的物理迭代次数。')
        self.solver_diagnostics = solver_diagnostics
        self.last_solver_diagnostics = None

    def forward(self, rx_iq, condition, region_ids, soft_tail, *, adapt_symbols, adapt_mask):
        corrected = self.model._apply_phase_correction(rx_iq, condition)
        if self.model.online_phase_trend_adapter is not None:
            corrected = self.model.online_phase_trend_adapter(corrected)
        rx = torch.complex(corrected[..., 0], corrected[..., 1])
        output = self.model._real_bpsk_known_adapt_logits(
            rx, condition, soft_tail, adapt_symbols, adapt_mask,
            return_solver_diagnostics=self.solver_diagnostics,
        )
        if self.solver_diagnostics:
            logits, self.last_solver_diagnostics = output
        else:
            logits = output
        return logits, logits.sigmoid()


class KnownAdaptRealNeuralBranch(nn.Module):
    """在相同 checkpoint 上启用实值已知 Adapt warm-start 与神经残差。"""

    def __init__(self, model):
        super().__init__()
        self.model = copy.deepcopy(model)
        self.model.config = replace(self.model.config, physics_real_bpsk_known_adapt=True)

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


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
