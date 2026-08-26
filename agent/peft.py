# -*- coding: utf-8 -*-
"""参数高效微调分组、快照与恢复工具。"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PEFTSnapshot:
    tensors: dict[str, torch.Tensor]


class PEFTRegistry:
    """统一管理线上可更新参数组。"""

    GROUP_ALIASES = {
        "conditioner_film": {"conditioner_film"},
        "adapter": {"adapter"},
        "attention_lora": {"attention_lora"},
        "ffn_lora": {"ffn_lora"},
        "head": {"head"},
        "adapter_lora": {"adapter", "attention_lora", "ffn_lora", "head"},
        "conditioner_peft": {"conditioner_film", "adapter", "attention_lora", "ffn_lora", "head"},
    }

    def __init__(self, module: nn.Module):
        self.module = module
        self._groups: dict[str, list[str]] = {name: [] for name in self.GROUP_ALIASES if name not in {"adapter_lora", "conditioner_peft"}}

    def register(self, group: str, module: nn.Module) -> None:
        if group not in self._groups:
            raise ValueError(f"未知 PEFT 组：{group}")
        for name, _ in module.named_parameters():
            self._groups[group].append(f"{module._get_name()}::{id(module)}::{name}")

    def resolve(self, groups: Iterable[str]) -> set[str]:
        resolved: set[str] = set()
        for group in groups:
            if group not in self.GROUP_ALIASES:
                raise ValueError(f"未知 PEFT 组：{group}")
            resolved.update(self.GROUP_ALIASES[group])
        return resolved

    def named_group_parameters(self, groups: Iterable[str]) -> list[tuple[str, nn.Parameter]]:
        resolved = self.resolve(groups)
        result = []
        for name, parameter in self.module.named_parameters():
            group = getattr(parameter, "_peft_group", None)
            if group in resolved:
                result.append((name, parameter))
        return result

    def parameters(self, groups: Iterable[str]) -> list[nn.Parameter]:
        return [parameter for _, parameter in self.named_group_parameters(groups)]

    def snapshot(self, groups: Iterable[str]) -> PEFTSnapshot:
        return PEFTSnapshot(
            {
                name: parameter.detach().clone()
                for name, parameter in self.named_group_parameters(groups)
            }
        )

    def restore(self, snapshot: PEFTSnapshot) -> None:
        named = dict(self.module.named_parameters())
        for name, value in snapshot.tensors.items():
            named[name].data.copy_(value)

    def delta_norm(self, snapshot: PEFTSnapshot) -> float:
        named = dict(self.module.named_parameters())
        total = torch.zeros((), dtype=torch.float32)
        for name, value in snapshot.tensors.items():
            total = total + torch.sum((named[name].detach().cpu() - value.cpu()) ** 2)
        return float(torch.sqrt(total).item())


def mark_peft_group(module: nn.Module, group: str) -> None:
    for parameter in module.parameters():
        setattr(parameter, "_peft_group", group)
