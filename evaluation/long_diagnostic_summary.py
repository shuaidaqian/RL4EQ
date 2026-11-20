# -*- coding: utf-8 -*-
"""从已完成逐帧评分派生诊断，不调用接收机或生成器。"""

import math

from scipy.special import ndtr


DERIVED_METRIC_FIELDS = (
    'cir_phase_only_squared_error_diagnostic',
    'pilot_innovation_energy_ratio_diagnostic',
    'full_energy_loose_genie_bound_diagnostic',
)


def derived_diagnostics(row):
    """先逐帧计算非线性量，再跨窗口平均，避免用平均值重建错误指标。

    下界使用事后仿真噪声和真实功率，只是理想辅助者的全能量宽松下界。
    未计算帧末截断；不得称为实际接收机或精确帧内下界。
    """

    truth_power = row['true_cir_power_diagnostic']
    true_noise = 10 ** (-row['snr_db'] / 10)
    predicted_variance = row['pilot_innovation_variance_mean']
    if truth_power < 0 or predicted_variance <= 0:
        raise ValueError('真实功率与预测能量不满足评分物理条件。')
    return dict(
        cir_phase_only_squared_error_diagnostic=row['phase_only_nmse_diagnostic'] * truth_power,
        pilot_innovation_energy_ratio_diagnostic=row['pilot_innovation_mse'] / predicted_variance,
        full_energy_loose_genie_bound_diagnostic=float(ndtr(-math.sqrt(2 * truth_power / true_noise))),
    )
