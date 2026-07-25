# -*- coding: utf-8 -*-
"""Continual PPO 新路线的通信环境占位模块。

Task 4 会在这里实现 acquisition、普通帧、receiver view 与跨帧 soft-tail。
当前阶段只保留可导入的公开边界，避免旧逐符号 A2C 环境继续引用已删除信道。
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentContract:
    """当前仓库重置阶段的环境契约标识。"""

    schema: str = "continual-ppo-env-v1"


def environment_schema() -> str:
    """返回新路线环境契约版本。"""

    return EnvironmentContract().schema
