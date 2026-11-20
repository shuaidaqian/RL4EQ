# -*- coding: utf-8 -*-
"""验证单帧 Pilot 对照与递归臂共享物理老化模型，但不累计普通帧 CIR 信息。"""

from types import SimpleNamespace

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from evaluation.continual_variants import validate_variants
import training.continual_receiver as receiver_module
from training.continual_receiver import ContinualReceiver, predict_cir_state, receiver_frame


def _acquisition():
    return SimpleNamespace(mean=torch.tensor([1. + .2j, .1 - .1j, .05j]),
        covariance=.2 * torch.eye(3, dtype=torch.complex64), noise_variance=.1, cfo=0.)


def _options(state_model, initial):
    return dict(state_model=state_model, process_noise=.03, state_rho=.8,
                stationary_power=[.6, .3, .1], initial_prediction_intervals=initial)


@pytest.mark.parametrize('state_model', ['random_walk', 'ar1'])
@pytest.mark.parametrize('initial', [0., 1., 3.5])
def test_actual_current_pilot_prior_is_aged_acquisition_not_previous_posterior(monkeypatch, state_model, initial):
    acquisition = _acquisition()
    options = _options(state_model, initial)
    receiver = ContinualReceiver(acquisition, torch.zeros(2, dtype=torch.complex64),
        arm='current_pilot', current_pilot_prior='aged_acquisition', **options)
    monkeypatch.setattr(receiver, '_phase_features', lambda frame, mean: torch.zeros(16))
    priors = []

    def record_and_change(frame, mean, covariance, noise, features):
        priors.append((mean.clone(), covariance.clone()))
        # 强制产生不同于先验的合法后验，下一帧对照仍须从 acquisition 重建。
        return mean + 3., covariance * .5

    monkeypatch.setattr(receiver_module, 'pilot_update', record_and_change)
    monkeypatch.setattr(receiver_module, 'context_logits', lambda *args, **kwargs: torch.zeros(128))
    generator = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=2), seed=31)
    for index in range(3):
        receiver.step(None, receiver_frame(generator.generate(index)))
        expected = predict_cir_state(acquisition.mean, acquisition.covariance,
            state_model=state_model, process_noise=.03, state_rho=.8,
            stationary_power=[.6, .3, .1], transition_intervals=initial + index)
        torch.testing.assert_close(priors[-1][0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(priors[-1][1], expected[1], rtol=0, atol=0)
    assert receiver.frames_seen == 3


@pytest.mark.parametrize('state_model', ['random_walk', 'ar1'])
def test_first_frame_aged_control_and_recursive_receiver_are_matched(state_model):
    torch.manual_seed(84)
    acquisition = _acquisition()
    options = _options(state_model, 3.5)
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=128, max_delay=2, iterations=1,
        d_model=8, num_heads=2, physics_warm_start_iterations=2)).eval()
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=2), seed=32).generate(0)
    from dataclasses import replace
    frame = receiver_frame(replace(frame, rx_symbols=torch.randn(128, dtype=torch.complex64)))
    control = ContinualReceiver(acquisition, torch.zeros(2, dtype=torch.complex64), arm='current_pilot',
        current_pilot_prior='aged_acquisition', phase_estimator='circular_clean', **options)
    recursive = ContinualReceiver(acquisition, torch.zeros(2, dtype=torch.complex64), arm='recursive_pilot',
        phase_estimator='circular_clean', **options)
    first, first_diagnostics = control.step(model, frame)
    second, second_diagnostics = recursive.step(model, frame)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first_diagnostics['decode_cir'], second_diagnostics['decode_cir'], rtol=0, atol=0)
    assert first_diagnostics['prior_covariance_trace'] == second_diagnostics['prior_covariance_trace']
    assert first_diagnostics['pilot_updates'] == second_diagnostics['pilot_updates'] == 1
    assert first_diagnostics['data_updates'] == second_diagnostics['data_updates'] == 0


@pytest.mark.parametrize('arm', ['acquisition', 'recursive_pilot', 'recursive_soft'])
def test_aged_control_cannot_silently_change_other_receiver_arms(arm):
    with pytest.raises(ValueError, match='只适用于'):
        ContinualReceiver(_acquisition(), torch.zeros(2), arm=arm, current_pilot_prior='aged_acquisition')
    with pytest.raises(ValueError, match='只适用于'):
        validate_variants(dict(variants=[dict(name='x', arm=arm, checkpoint='example.pt',
            current_pilot_prior='aged_acquisition')]))


def test_variant_manifest_explicitly_accepts_and_checks_current_pilot_prior():
    variant = dict(name='aged', arm='current_pilot', checkpoint='example.pt',
        current_pilot_prior='aged_acquisition')
    assert validate_variants(dict(variants=[variant])) == [variant]
    with pytest.raises(ValueError, match='未知当前'):
        validate_variants(dict(variants=[dict(variant, current_pilot_prior='unknown')]))
    with pytest.raises(ValueError, match='当前 Pilot 先验'):
        ContinualReceiver(_acquisition(), torch.zeros(2), arm='current_pilot', current_pilot_prior='unknown')
