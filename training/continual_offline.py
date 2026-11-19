# -*- coding: utf-8 -*-
"""离线闭环训练：先用冻结模型完成接收轨迹，再训练分离的 Data 监督。"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace

import torch
import torch.nn.functional as F
from torch import nn


def detached_copy(value):
    """快照实际前向输入，避免后续接收状态或训练更新改变轨迹样本。"""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{field.name: detached_copy(getattr(value, field.name)) for field in fields(value)})
    if isinstance(value, dict):
        return {key: detached_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(detached_copy(item) for item in value)
    return value


@dataclass(frozen=True)
class OfflineExample:
    """真实推理输入与仅离线使用的监督目标，分别存储。"""
    args: tuple
    kwargs: dict
    labels: torch.Tensor
    mask: torch.Tensor
    frame_index: int


class InputRecorder(nn.Module):
    """透明转发并快照一次实际前向；不重建或猜测物理接收上下文。"""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.last_inputs = None

    @property
    def config(self):
        return self.model.config

    def forward(self, *args, **kwargs):
        self.last_inputs = detached_copy((args, kwargs))
        return self.model(*args, **kwargs)

    def example(self, labels, mask, frame_index):
        if self.last_inputs is None:
            raise RuntimeError('尚无实际接收前向，不能创建离线训练样本。')
        args, kwargs = self.last_inputs
        return OfflineExample(args, kwargs, labels.detach().clone(), mask.detach().clone(), int(frame_index))


def train_examples(model, optimizer, examples, *, order, max_grad_norm=1.0):
    """轨迹完成后逐样本训练；不通过长序列滤波或历史 soft-tail 反传。"""
    if not examples or sorted(order) != list(range(len(examples))):
        raise ValueError('训练样本必须非空，且顺序须覆盖全部样本一次。')
    # 保持与部署一致的 eval 路径；eval 不禁止梯度，监督仅存在于本函数。
    model.eval()
    rows = []
    for index in order:
        example = examples[index]
        if not bool(example.mask.any()):
            raise ValueError('离线 Data 监督掩码不能为空。')
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(*example.args, **example.kwargs)
        selected = logits.squeeze(0)[example.mask]
        loss = F.binary_cross_entropy_with_logits(selected, example.labels[example.mask].float())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('离线闭环训练损失非有限，停止并保留失败记录。')
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        rows.append(dict(frame_index=example.frame_index, data_bce=float(loss.detach()),
                         data_bits=int(example.mask.sum()), gradient_norm=float(grad_norm)))
    return rows
