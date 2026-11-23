# -*- coding: utf-8 -*-
"""验证合法 Pilot 诊断的物理量、标签隔离和流式保存完整性。"""

from dataclasses import replace
import json
from types import SimpleNamespace

import pytest
import torch

from env.frame_structure import FrameConfig, FrameGenerator
from scripts.compare_continual_variants import STREAM_ROW_FIELDS, compact_stream_row
from training.continual_receiver import pilot_innovation_statistics, receiver_frame
import training.continual_receiver as receiver_module


def _known_pilot_frame(*, frame_len=128, max_delay=2, total_pilot=64, reward=16):
    frame = FrameGenerator(FrameConfig(frame_len=frame_len, max_delay=max_delay,
        total_pilot=total_pilot, reward_pilot_total=reward), seed=61).generate(0)
    symbols = torch.ones(frame_len, dtype=torch.complex64)
    mean = torch.ones(max_delay + 1, dtype=torch.complex64) / (max_delay + 1)
    rx = torch.full((frame_len,), 1.3 + .4j, dtype=torch.complex64)
    return replace(frame, tx_symbols=symbols, rx_symbols=rx), mean


def test_pilot_statistics_use_complex_residual_power_and_full_prior_covariance():
    frame, mean = _known_pilot_frame()
    covariance = (.01 * torch.ones(3, 3) + .02 * torch.eye(3)).to(torch.complex64)
    result = pilot_innovation_statistics(receiver_frame(frame), mean, covariance, .2, torch.zeros(2))
    assert result['pilot_observations'] == 46
    assert result['pilot_innovation_mse'] == pytest.approx(.3**2 + .4**2)
    # X 的每行是 [1,1,1]，包含六个非对角项，不能只使用 P 的迹。
    assert result['pilot_innovation_variance_mean'] == pytest.approx(.2 + .01 * 9 + .02 * 3)


def test_fixed_research_frame_has_108_clean_adapt_observations():
    frame, mean = _known_pilot_frame(frame_len=1024, max_delay=116, total_pilot=256, reward=32)
    covariance = torch.eye(117, dtype=torch.complex64) * .01
    result = pilot_innovation_statistics(receiver_frame(frame), mean, covariance, .2, torch.zeros(2))
    assert result['pilot_observations'] == 224 - 116
    assert result['pilot_innovation_variance_mean'] == pytest.approx(.2 + 117 * .01)


def test_pilot_statistics_ignore_hidden_labels_truth_and_nonadapt_samples():
    frame, mean = _known_pilot_frame()
    covariance = torch.eye(3, dtype=torch.complex64) * .01
    poisoned_symbols = frame.tx_symbols.clone()
    poisoned_symbols[~frame.adapt_mask] = 999 + 999j
    poisoned_rx = frame.rx_symbols.clone()
    poisoned_rx[~frame.adapt_mask] = -1000j
    poisoned = replace(frame, bits=~frame.bits, tx_symbols=poisoned_symbols, rx_symbols=poisoned_rx,
                       true_cir=torch.ones(3) * 1000, tail_symbols=torch.ones(2) * 1000)
    first = pilot_innovation_statistics(receiver_frame(frame), mean, covariance, .2, torch.zeros(2))
    second = pilot_innovation_statistics(receiver_frame(poisoned), mean, covariance, .2, torch.zeros(2))
    assert first == second


def test_empty_pilot_statistics_are_explicit_nulls_instead_of_nan():
    frame, mean = _known_pilot_frame()
    frame = replace(frame, adapt_mask=torch.zeros_like(frame.adapt_mask))
    result = pilot_innovation_statistics(receiver_frame(frame), mean, torch.eye(3), .2, torch.zeros(2))
    assert result == {'pilot_observations': 0, 'pilot_innovation_mse': None,
                      'pilot_innovation_variance_mean': None}
    json.dumps(result, allow_nan=False)


def test_stream_roundtrip_preserves_covariance_and_legal_observation_diagnostics():
    row = dict.fromkeys(STREAM_ROW_FIELDS, 1)
    row.update(arm='candidate', rx_sha256='waveform', pilot_innovation_mse=.25,
               pilot_innovation_variance_mean=.35, prior_covariance_trace=.12,
               decode_covariance_trace=.04, posterior_covariance_trace=.03,
               true_cir=torch.ones(3), data_labels=torch.ones(3), shared_state_from='source')
    saved = json.loads(json.dumps(compact_stream_row(row), allow_nan=False))
    assert saved['prior_covariance_trace'] == .12
    assert saved['decode_covariance_trace'] == .04
    assert saved['posterior_covariance_trace'] == .03
    assert saved['pilot_innovation_mse'] == .25
    assert saved['pilot_innovation_variance_mean'] == .35
    assert saved['shared_state_from'] == 'source'
    assert 'true_cir' not in saved and 'data_labels' not in saved


def test_receiver_diagnostics_reach_jsonl_schema_without_reconstruction(monkeypatch):
    frame, mean = _known_pilot_frame()
    acquisition = SimpleNamespace(mean=mean, covariance=torch.eye(3, dtype=torch.complex64) * .1,
                                  noise_variance=.2, cfo=0.)
    receiver = receiver_module.ContinualReceiver(acquisition, torch.zeros(2), arm='recursive_pilot')
    monkeypatch.setattr(receiver, '_phase_features', lambda frame, state: torch.zeros(2))
    monkeypatch.setattr(receiver_module, 'context_logits', lambda *args, **kwargs: torch.zeros(128))
    _, diagnostics = receiver.step(None, receiver_frame(frame))
    diagnostics.pop('decode_cir')
    row = dict(seed=1, snr_db=0, frame_index=0, arm='candidate', errors=0, bits=64,
               elapsed_ms=1., rx_sha256='paired', noise_variance_used=.2, model_calls=1,
               true_cir_norm_diagnostic=1., **diagnostics)
    saved = json.loads(json.dumps(compact_stream_row(row), allow_nan=False))
    assert saved['pilot_observations'] == 46
    assert saved['pilot_updates'] == 1
    assert saved['prior_covariance_trace'] > saved['decode_covariance_trace']
    assert saved['posterior_covariance_trace'] == saved['decode_covariance_trace']
    assert saved['pilot_innovation_mse'] == pytest.approx(.25)


@pytest.mark.parametrize('missing', ['posterior_covariance_trace', 'pilot_innovation_mse'])
def test_stream_rejects_missing_diagnostics_instead_of_silently_dropping_them(missing):
    row = dict.fromkeys(STREAM_ROW_FIELDS, 1)
    del row[missing]
    with pytest.raises(ValueError, match='缺少必要字段'):
        compact_stream_row(row)
