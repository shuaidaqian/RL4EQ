# -*- coding: utf-8 -*-
"""尾部错误集中度不强行打破并列、不误将空错误分母变成零结论。"""

import numpy as np
import pytest

from scripts.analyze_completed_diagnostic_tails import summarize_distribution


def test_tied_diagnostic_tail_keeps_actual_fraction_and_error_concentration():
    values = [1.] * 90 + [2.] * 10
    errors = [0] * 90 + [1] * 10
    summary = summarize_distribution(values, errors, [100] * 100, bad_direction='high')
    five_percent = summary['tails'][1]
    assert five_percent['nominal_fraction'] == .05
    assert five_percent['frames'] == 10
    assert five_percent['frame_fraction'] == .1
    assert five_percent['error_fraction'] == 1.
    assert five_percent['conditional_ber'] == .01


def test_low_tail_and_no_errors_are_not_reported_as_no_effect():
    summary = summarize_distribution(np.arange(100), [0] * 100, [100] * 100, bad_direction='low')
    assert summary['tails'][1]['frames'] == 5
    assert summary['tails'][1]['threshold'] < 5
    assert summary['tails'][1]['error_fraction'] is None
    assert summarize_distribution([1.], [1], [100], bad_direction='high')['tails'] == []


def test_invalid_diagnostic_is_not_silently_dropped():
    with pytest.raises(ValueError):
        summarize_distribution([1., float('nan')], [0, 0], [1, 1], bad_direction='high')
