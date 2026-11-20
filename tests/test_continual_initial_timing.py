# -*- coding: utf-8 -*-
"""从实际传入 Pilot 更新的先验验证首帧及后续帧的物理时序。"""

from types import SimpleNamespace

import pytest
import torch

from env.frame_structure import FrameConfig, FrameGenerator
import training.continual_receiver as receiver_module
from training.continual_receiver import acquisition_transition_intervals, predict_cir_state


@pytest.mark.parametrize("gap,expected", [(0.0, 1.0), (0.256, 3.5), (0.1024, 2.0)])
def test_acquisition_interval_includes_frame_end_transition_and_extra_gap(gap, expected):
    assert acquisition_transition_intervals(0.1024, gap) == pytest.approx(expected)


@pytest.mark.parametrize("duration,gap", [(0, 0), (-1, 0), (float('nan'), 0),
                                           (1, -1), (1, float('inf'))])
def test_acquisition_interval_rejects_invalid_physical_time(duration, gap):
    with pytest.raises(ValueError):
        acquisition_transition_intervals(duration, gap)


def _expected_prior(mean, covariance, *, model, intervals):
    if model == 'random_walk':
        return mean, covariance + .03 * intervals * torch.eye(3, dtype=covariance.dtype)
    coefficient = .8 ** intervals
    stationary = torch.diag(torch.tensor([.6, .3, .1], dtype=covariance.dtype))
    return coefficient * mean, coefficient**2 * covariance + (1 - coefficient**2) * stationary


@pytest.mark.parametrize("state_model", ['random_walk', 'ar1'])
@pytest.mark.parametrize("arm", ['acquisition', 'current_pilot', 'recursive_pilot', 'recursive_soft'])
@pytest.mark.parametrize("initial_intervals", [None, 0.0, 1.0, 3.5])
def test_first_pilot_prior_accounts_for_initial_interval_exactly_once(
        monkeypatch, state_model, arm, initial_intervals):
    mean = torch.tensor([1 + .5j, .2 - .1j, .1j], dtype=torch.complex64)
    covariance = torch.eye(3, dtype=torch.complex64) * .2
    acquisition = SimpleNamespace(mean=mean, covariance=covariance, noise_variance=.1, cfo=0.)
    options = {} if initial_intervals is None else {'initial_prediction_intervals': initial_intervals}
    receiver = receiver_module.ContinualReceiver(
        acquisition, torch.zeros(2, dtype=torch.complex64), arm=arm,
        process_noise=.03, state_model=state_model, state_rho=.8,
        stationary_power=[.6, .3, .1], **options,
    )
    # 隔离观测更新与解码，只记录实际接收到的先验；不读取信道真值或标签。
    monkeypatch.setattr(receiver, '_phase_features', lambda frame, state: torch.zeros(2))
    priors = []

    def record_prior(frame, state, cov, noise, features):
        priors.append((state.clone(), cov.clone()))
        return state, cov

    monkeypatch.setattr(receiver_module, 'pilot_update', record_prior)
    monkeypatch.setattr(receiver_module, 'soft_update', lambda frame, logits, state, cov, noise, features: (state, cov))
    monkeypatch.setattr(receiver_module, 'context_logits', lambda *args, **kwargs: torch.zeros(128))
    monkeypatch.setattr(receiver_module, 'tail_from_logits', lambda logits, tail: tail)
    generator = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=2), seed=42)
    for index in range(2):
        receiver.step(None, receiver_module.receiver_frame(generator.generate(index)))
        if arm == 'acquisition':
            # Frozen 没有 Pilot 更新且保持 acquisition 状态。
            assert priors == []
            actual = receiver.mean, receiver.covariance
            expected = mean, covariance
        elif arm == 'current_pilot':
            actual = priors[index]
            expected = mean, covariance
        else:
            actual = priors[index]
            start = 1.0 if initial_intervals is None else initial_intervals
            expected = _expected_prior(mean, covariance, model=state_model, intervals=start + index)
        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize("state_model", ['random_walk', 'ar1'])
def test_fractional_transition_composes_without_duplicate_process_noise(state_model):
    mean = torch.tensor([1 + 1j, 2 - 1j], dtype=torch.complex64)
    covariance = torch.tensor([[.5, .1j], [-.1j, .3]], dtype=torch.complex64)
    options = dict(state_model=state_model, process_noise=.02, state_rho=.7,
                   stationary_power=[.6, .4])
    first = predict_cir_state(mean, covariance, transition_intervals=1, **options)
    split = predict_cir_state(*first, transition_intervals=2.5, **options)
    combined = predict_cir_state(mean, covariance, transition_intervals=3.5, **options)
    torch.testing.assert_close(split[0], combined[0])
    torch.testing.assert_close(split[1], combined[1])


@pytest.mark.parametrize("invalid", [-1, float('nan'), float('inf')])
def test_prediction_rejects_invalid_transition_interval(invalid):
    with pytest.raises(ValueError, match='转移间隔'):
        predict_cir_state(torch.ones(2, dtype=torch.complex64), torch.eye(2),
                          state_model='random_walk', process_noise=.01,
                          transition_intervals=invalid)
