# -*- coding: utf-8 -*-
"""防止在时间窗口均值上误算绝对状态误差、能量比或非线性物理下界。"""

import pytest
from scipy.special import ndtr

from evaluation.long_diagnostic_summary import derived_diagnostics


def _row(power, nmse, residual, variance, snr=0):
    return dict(true_cir_power_diagnostic=power, phase_only_nmse_diagnostic=nmse,
        pilot_innovation_mse=residual, pilot_innovation_variance_mean=variance, snr_db=snr)


def test_absolute_cir_error_is_computed_per_frame_before_averaging():
    first = derived_diagnostics(_row(1., .2, 2., 1.))
    second = derived_diagnostics(_row(3., .4, 2., 4.))
    assert (first['cir_phase_only_squared_error_diagnostic'] +
            second['cir_phase_only_squared_error_diagnostic']) / 2 == pytest.approx(.7)
    # 平均 NMSE 乘平均功率为 0.6，不是上面的实际平均平方误差。
    assert .7 != pytest.approx((.2 + .4) / 2 * (1. + 3.) / 2)


def test_mean_energy_ratio_is_not_ratio_of_mean_energies():
    first = derived_diagnostics(_row(1., .2, 2., 1.))
    second = derived_diagnostics(_row(3., .4, 2., 4.))
    mean_ratio = (first['pilot_innovation_energy_ratio_diagnostic'] +
                  second['pilot_innovation_energy_ratio_diagnostic']) / 2
    assert mean_ratio == 1.25
    assert mean_ratio != pytest.approx(2 / 2.5)


def test_genie_bound_uses_true_noise_and_is_computed_before_window_averaging():
    first = derived_diagnostics(_row(.5, 0., 1., 1.))
    second = derived_diagnostics(_row(1.5, 0., 1., 1.))
    assert first['full_energy_loose_genie_bound_diagnostic'] == pytest.approx(ndtr(-1.))
    mean_bound = (first['full_energy_loose_genie_bound_diagnostic'] +
                  second['full_energy_loose_genie_bound_diagnostic']) / 2
    assert mean_bound > ndtr(-(2**.5))


def test_invalid_prediction_energy_is_not_silently_repaired():
    with pytest.raises(ValueError):
        derived_diagnostics(_row(1., .1, 1., 0.))
