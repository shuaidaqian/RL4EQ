# -*- coding: utf-8 -*-
"""加载并验证可复用的离线 CIR 动态统计。"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def load_ar1_dynamics(path, *, expected_sha256=None, tap_count=None):
    """读取冻结 AR(1) 参数并验证来源、有限性与 CIR 维度。"""

    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"AR(1) 动态文件指纹不匹配：{source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    rho = payload.get("rho_estimate")
    power = payload.get("stationary_power_by_tap")
    if not isinstance(rho, (int, float)) or not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("AR(1) rho_estimate 必须为 [0,1) 内有限数。")
    if not isinstance(power, list) or not power:
        raise ValueError("AR(1) stationary_power_by_tap 必须为非空列表。")
    if tap_count is not None and len(power) != tap_count:
        raise ValueError("AR(1) stationary_power_by_tap 长度与 CIR 维度不符。")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in power):
        raise ValueError("AR(1) stationary_power_by_tap 必须为非负有限数。")
    return {"rho_estimate": float(rho),
            "stationary_power_by_tap": [float(value) for value in power],
            "sha256": digest}
