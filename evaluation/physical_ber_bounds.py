# -*- coding: utf-8 -*-
"""当前帧观测下的 BPSK 额外信息 BER 下界，仅供事后物理分析。

辅助者告知真实信道、逐样本相位、前史与其余符号，保留待判比特未知。
该下界不能作为合法接收机的实测 BER，也不能进入在线更新。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr


def _validate(tap_values, data_mask, noise_variance):
    values = np.asarray(tap_values)
    mask = np.asarray(data_mask)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError('tap 向量必须非空、一维且有限。')
    if mask.ndim != 1 or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('Data mask 必须为一维布尔向量，且至少包含一个比特。')
    if values.size > mask.size:
        raise ValueError('当前实现要求帧长不小于 CIR 长度。')
    if isinstance(noise_variance, (bool, np.bool_)) or not math.isfinite(float(noise_variance)) or noise_variance <= 0:
        raise ValueError('复噪声功率必须为正有限数。')
    positions = np.flatnonzero(mask)
    # 位置 i 的符号仅在已接收的 i...N-1 出现，不补未来帧观测。
    last_tap = np.minimum(values.size - 1, mask.size - 1 - positions)
    return values, mask, last_tap, float(noise_variance)


def conditional_frame_genie_bound(cir, data_mask, noise_variance):
    """固定真实 CIR 后，计算包含帧末列截断的平均单比特检测下界。

    复噪声定义为 E|n|²=σ²，每个实分量的方差为 σ²/2。
    即使列无观测能量也返回合法的 0.5，不用数值地板虚构信息。
    """

    values, mask, last_tap, noise = _validate(cir, data_mask, noise_variance)
    power = np.abs(values.astype(np.complex128)) ** 2
    prefix_energy = np.cumsum(power)
    energy = prefix_energy[last_tap]
    per_bit = ndtr(-np.sqrt(2 * energy / noise))
    full_column = last_tap == values.size - 1
    return dict(
        data_bits=int(mask.sum()), full_column_bits=int(full_column.sum()),
        truncated_column_bits=int((~full_column).sum()),
        genie_ber_bound=float(per_bit.mean()),
        full_energy_loose_bound=float(ndtr(-math.sqrt(2 * float(prefix_energy[-1]) / noise))),
        full_column_bound_contribution=float(per_bit[full_column].sum() / per_bit.size),
        truncated_column_bound_contribution=float(per_bit[~full_column].sum() / per_bit.size),
        mean_observed_column_energy=float(energy.mean()),
    )


def expected_fading_frame_genie_bound(base_tap_power, data_mask, noise_variance, *, elapsed_over_coherence):
    """条件于基准 PDP，在生成器 AR(1) 分布下积分平均帧内下界。

    初始 |h_j|²=p_j，创新协方差为 diag(p)。不把接收机离线平均 D
    自动当作某个实际会话的基准 PDP，也不生成随机轨迹或使用 seed。
    """

    values, _, last_tap, noise = _validate(base_tap_power, data_mask, noise_variance)
    if np.iscomplexobj(values) or np.any(values < 0):
        raise ValueError('基准 tap 功率必须为非负实数。')
    elapsed = float(elapsed_over_coherence)
    if not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError('归一化演化时间必须非负且有限。')
    power = values.astype(np.float64)
    prefix_energy = np.cumsum(power)
    if elapsed == 0 or prefix_energy[-1] == 0:
        return dict(genie_ber_bound=float(ndtr(-np.sqrt(2 * prefix_energy[last_tap] / noise)).mean()),
                    quadrature_error=0.)
    indices, counts = np.unique(last_tap, return_counts=True)
    weights = counts / last_tap.size
    a = math.exp(-2 * elapsed)
    v = -math.expm1(-2 * elapsed)

    def integrand(theta):
        sin2 = math.sin(theta) ** 2
        if sin2 == 0:
            return float(weights[prefix_energy[indices] == 0].sum() / math.pi)
        coefficients = power / (noise * sin2)
        denominator = 1 + coefficients * v
        log_terms = -np.log1p(coefficients * v) - coefficients * a / denominator
        prefix_laplace = np.exp(np.cumsum(log_terms)[indices])
        return float(weights @ prefix_laplace / math.pi)

    value, error = quad(integrand, 0, math.pi / 2, epsabs=1e-13, epsrel=1e-10)
    return dict(genie_ber_bound=float(value), quadrature_error=float(error))
