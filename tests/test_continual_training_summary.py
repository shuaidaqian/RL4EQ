# -*- coding: utf-8 -*-
"""交互符号与训练重复/接收会话统计单位的检查。"""

import pytest

from evaluation.continual_training_summary import summarize_training_interaction


def interaction_fixture():
    variants, rows = [], []
    for repeat in range(3):
        for method in ['acquisition', 'recursive_pilot']:
            for arm in ['current_pilot', 'recursive_pilot']:
                name = f'{repeat}_{method}_{arm}'
                variants.append(dict(name=name, training_repeat=repeat, training_method=method, arm=arm))
                errors = (30 if arm == 'current_pilot' else 20) if method == 'acquisition' else (25 if arm == 'current_pilot' else 10)
                for seed in [1, 2]:
                    for frame in range(8):
                        rows.append(dict(snr_db=0, seed=seed, frame_index=frame, arm=name, errors=errors,
                            bits=100, rx_sha256=f'{seed}_{frame}', elapsed_ms=1))
    return dict(status='completed', protocol=dict(frames=8, snrs=[0], seeds=[1, 2],
        arms=[v['name'] for v in variants], variants=variants, pairs=[]), rows=rows,
        acquisitions=[dict(snr_db=0, noise_estimate=1., noise_truth_diagnostic=1.)])


def test_interaction_uses_two_crossed_axes_and_correct_sign():
    row = summarize_training_interaction(interaction_fixture())['per_snr']['0']
    assert row['mean_interaction'] == pytest.approx(.05)
    assert row['by_receiver_session_conditioning_on_trained_models']['sessions'] == 2
    assert row['by_training_repeat_conditioning_on_receiver_sessions']['sessions'] == 3
    assert row['joint_generalization_ci'] is None


def test_incomplete_or_different_waveforms_cannot_support_training_interaction():
    result = interaction_fixture()
    result['status'] = 'interrupted'
    with pytest.raises(ValueError, match='未完成'):
        summarize_training_interaction(result)
    result['status'] = 'completed'
    result['rows'][0]['rx_sha256'] = 'different'
    with pytest.raises(ValueError, match='波形'):
        summarize_training_interaction(result)
