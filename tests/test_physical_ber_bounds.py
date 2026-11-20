# -*- coding: utf-8 -*-
"""用可精确求解的边界核验物理下界，防止把弱参照当作不可能性证据。"""

import math

import numpy as np
import pytest
from scipy.special import ndtr

from evaluation.physical_ber_bounds import conditional_frame_genie_bound, expected_fading_frame_genie_bound


def test_memoryless_bpsk_uses_complex_noise_power_and_matches_awgn():
    result = conditional_frame_genie_bound([1j], np.ones(8, dtype=bool), .5)
    assert result['genie_ber_bound'] == pytest.approx(ndtr(-2.))
    assert result['genie_ber_bound'] == result['full_energy_loose_bound']
    assert result['truncated_column_bits'] == 0


def test_unobserved_final_delayed_bit_has_one_half_error_even_without_interference():
    result = conditional_frame_genie_bound([0., 1.], np.ones(4, dtype=bool), .01)
    assert result['truncated_column_bits'] == 1
    assert result['genie_ber_bound'] == pytest.approx(.125)
    assert result['full_energy_loose_bound'] < 1e-40


def test_fixed_frame_has_652_full_columns_and_116_truncated_data_columns():
    mask = np.arange(1024) >= 256
    result = conditional_frame_genie_bound(np.ones(117) / math.sqrt(117), mask, .1)
    assert result['data_bits'] == 768
    assert result['full_column_bits'] == 652
    assert result['truncated_column_bits'] == 116
    assert result['genie_ber_bound'] > result['full_energy_loose_bound']
    assert result['genie_ber_bound'] == pytest.approx(
        result['full_column_bound_contribution'] + result['truncated_column_bound_contribution'])


def test_phase_and_joint_signal_noise_scaling_do_not_change_bound():
    mask = np.arange(12) >= 4
    cir = np.array([.8 + .2j, .3 - .1j, -.2j])
    baseline = conditional_frame_genie_bound(cir, mask, .2)
    rotated_scaled = conditional_frame_genie_bound(3 * np.exp(1.4j) * cir, mask, 9 * .2)
    assert rotated_scaled['genie_ber_bound'] == pytest.approx(baseline['genie_ber_bound'])


def test_zero_energy_returns_one_half_instead_of_artificial_information():
    mask = np.ones(4, dtype=bool)
    assert conditional_frame_genie_bound([0., 0.], mask, 1.)['genie_ber_bound'] == .5
    assert expected_fading_frame_genie_bound([0., 0.], mask, 1., elapsed_over_coherence=1.)['genie_ber_bound'] == .5


def test_fading_initial_value_matches_conditional_bound_with_frame_truncation():
    power = np.array([.5, .3, .2])
    mask = np.arange(10) >= 4
    expected = expected_fading_frame_genie_bound(power, mask, .3, elapsed_over_coherence=0.)
    fixed = conditional_frame_genie_bound(np.sqrt(power), mask, .3)
    assert expected['genie_ber_bound'] == pytest.approx(fixed['genie_ber_bound'])


def test_single_tap_stationary_limit_matches_rayleigh_closed_form():
    noise = .2
    result = expected_fading_frame_genie_bound([1.], np.ones(4, dtype=bool), noise, elapsed_over_coherence=20.)
    assert result['genie_ber_bound'] == pytest.approx(.5 * (1 - math.sqrt(1 / (1 + noise))), abs=1e-12)


@pytest.mark.parametrize('noise', [1., 10**(-.5), .1, 10**(-1.5)])
def test_expected_finite_frame_bound_is_nondecreasing_as_fading_distribution_develops(noise):
    values = [expected_fading_frame_genie_bound([.7, .2, .1], np.arange(12) >= 4, noise,
              elapsed_over_coherence=t)['genie_ber_bound'] for t in [0., .01, .1, .5, 1., 2., 10.]]
    assert np.all(np.diff(values) > 0)


@pytest.mark.parametrize('power,mask,noise', [([1.], [True], 0), ([1.], [True], True),
    ([np.nan], [True], 1.), ([1.], [False], 1.), ([1.], [1], 1.), ([1., 2.], [True], 1.)])
def test_invalid_physical_inputs_are_rejected(power, mask, noise):
    with pytest.raises(ValueError):
        conditional_frame_genie_bound(power, mask, noise)


def test_fading_model_rejects_negative_powers_and_negative_time():
    with pytest.raises(ValueError):
        expected_fading_frame_genie_bound([-.1, 1.1], np.ones(8, dtype=bool), 1., elapsed_over_coherence=1.)
    with pytest.raises(ValueError):
        expected_fading_frame_genie_bound([1.], np.ones(8, dtype=bool), 1., elapsed_over_coherence=-1.)
