# -*- coding: utf-8 -*-
"""防止长程汇总把 Frozen 退化当自身改善、漏掉不利会话或错误平均包裹相位。"""

import copy

import pytest

from evaluation.long_diagnostic_summary import paired_window_summary


def _fixture():
    manifest = dict(variants=[dict(name='frozen'), dict(name='online')], frames=4,
        seeds=[1, 2], snrs=[0], pairs=[['frozen', 'online']], pre_registered_windows=[[0, 2], [2, 4]])
    rows = []
    for seed in manifest['seeds']:
        for arm, rates in [('frozen', [.1, .4]), ('online', [.08, .12])]:
            for scope, window, rate in [('registered_window', [0, 2], rates[0]),
                ('registered_window', [2, 4], rates[1]), ('whole_session', [0, 4], sum(rates) / 2)]:
                frames = window[1] - window[0]
                rows.append(dict(snr_db=0, seed=seed, arm=arm, scope=scope, window=window,
                    ber=rate, frames=frames, errors=round(rate * frames * 1000), bits=frames * 1000,
                    means=dict(true_cir_power_diagnostic=float(seed),
                        decoder_phase_data_mean_rad_diagnostic=3.1 if seed == 1 else -3.1)))
    return dict(status='completed_development_audit', session_summary=rows), manifest


def test_gain_growth_keeps_online_degradation_visible_with_correct_identity():
    audit, manifest = _fixture()
    summary = paired_window_summary(audit, manifest)
    assert len(summary['adjacent_window_changes']) == 2
    for row in summary['adjacent_window_changes']:
        assert row['delta_online_ber'] == pytest.approx(.04)
        assert row['delta_baseline_ber'] == pytest.approx(.3)
        assert row['delta_gain'] == pytest.approx(.26)
        assert row['delta_gain'] == row['delta_baseline_ber'] - row['delta_online_ber']


def test_all_registered_windows_and_sessions_are_required_before_pairing():
    audit, manifest = _fixture()
    audit['session_summary'].pop()
    with pytest.raises(ValueError, match='不完整'):
        paired_window_summary(audit, manifest)


def test_duplicated_session_windows_are_not_treated_as_extra_independent_samples():
    audit, manifest = _fixture()
    audit['session_summary'].append(copy.deepcopy(audit['session_summary'][0]))
    with pytest.raises(ValueError, match='重复'):
        paired_window_summary(audit, manifest)


def test_wrapped_phase_means_are_not_arithmetically_averaged_across_sessions():
    audit, manifest = _fixture()
    summary = paired_window_summary(audit, manifest)
    for row in summary['aggregate_windows']:
        assert row['sessions'] == 2
        assert row['means']['true_cir_power_diagnostic'] == 1.5
        assert 'decoder_phase_data_mean_rad_diagnostic' not in row['means']


def test_incomplete_experiment_cannot_generate_performance_summary():
    audit, manifest = _fixture()
    audit['status'] = 'running'
    with pytest.raises(ValueError, match='完整审计'):
        paired_window_summary(audit, manifest)
