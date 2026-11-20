# -*- coding: utf-8 -*-
"""验证事后同步评分的相位规范、实际解码校正、流式保存及不回流约束。"""

from dataclasses import replace
import json

import pytest
import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from env.frame_structure import FrameConfig, FrameGenerator
from env.linear_operator import LinearChannelOperator
from evaluation.continual_variants import ForwardCounter, KnownAdaptRealPhysicsBranch
from evaluation.receiver_diagnostics import (
    LONG_DIAGNOSTIC_FIELDS, RECEIVER_SCORE_FIELDS, long_receiver_diagnostics, receiver_diagnostics,
)
from scripts.compare_continual_variants import STREAM_ROW_FIELDS, compact_stream_row
from training.continual_offline import InputRecorder
from training.continual_receiver import ContinualReceiver, make_receiver_context, receiver_frame
from tests.test_continual_receiver import acquisition_fixture
from training.continual_receiver import estimate_acquisition


def _fixture(*, gauge=.7, amplitude=1., cfo=.001, phase_offset=.3, learned_offset=.2, scale=.8):
    torch.manual_seed(71)
    frame = FrameGenerator(FrameConfig(frame_len=128, total_pilot=64, max_delay=2), seed=73).generate(0)
    truth = torch.tensor([.9 + .1j, .2 - .1j, .1j])
    tail = torch.ones(2, dtype=torch.complex64)
    phase = phase_offset + 2 * torch.pi * cfo * torch.arange(128)
    noiseless = LinearChannelOperator(128, 2, backend='fft').forward(frame.tx_symbols, truth, tail)
    frame = replace(frame, true_cir=truth, tail_symbols=tail, rx_symbols=noiseless * torch.exp(1j * phase))
    estimate = amplitude * torch.exp(torch.tensor(1j * gauge)) * truth
    features = torch.zeros(16)
    # 补偿 CIR 中吸收的相位，并反解已学习偏移和缩放，使实际预处理恰好正确。
    features[0::4] = (phase_offset - gauge) / scale - learned_offset
    features[1::4] = cfo / scale
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=128, max_delay=2, iterations=1,
        d_model=8, num_heads=2, physics_warm_start_iterations=2,
        enable_phase_correction_branch=True, phase_correction_initial_scale=scale)).eval()
    with torch.no_grad():
        model.phase_correction[-1].bias.fill_(learned_offset)
    context = make_receiver_context(receiver_frame(frame), estimate, tail, .1, features)
    iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), -1)[None]
    actual = ((iq, context.condition, frame.model_region_ids[None], tail), {})
    return model, actual, frame, phase, cfo


@pytest.mark.parametrize('gauge', [-2., .7, 2.9])
def test_phase_score_respects_cir_gauge_and_learned_decoder_correction(gauge):
    model, actual, frame, phase, cfo = _fixture(gauge=gauge)
    result = long_receiver_diagnostics(KnownAdaptRealPhysicsBranch(model), actual, frame, phase, cfo, .1)
    assert result['decoder_phase_data_rms_rad_diagnostic'] < 1e-5
    assert result['decoder_signal_mismatch_nmse_diagnostic'] < 1e-10
    assert result['decoder_effective_cfo_estimate_diagnostic'] == pytest.approx(cfo, abs=1e-8)
    assert result['realized_data_noise_power_diagnostic'] < 1e-10


def test_waveform_score_exposes_amplitude_mismatch_even_with_exact_phase():
    model, actual, frame, phase, cfo = _fixture(amplitude=2.)
    result = long_receiver_diagnostics(model, actual, frame, phase, cfo, .1)
    assert result['decoder_phase_data_rms_rad_diagnostic'] < 1e-5
    assert result['decoder_signal_mismatch_nmse_diagnostic'] == pytest.approx(1., abs=1e-5)


def test_actual_receiver_and_stream_keep_scores_without_truth_feedback():
    torch.manual_seed(72)
    model, _, frame, phase, cfo = _fixture()
    tx, rx, _ = acquisition_fixture()
    acquisition = estimate_acquisition(tx, rx, 2, cfo_limit=0)
    first = ContinualReceiver(acquisition, torch.zeros(2, dtype=torch.complex64),
        arm='recursive_pilot', phase_estimator='circular_clean')
    second = ContinualReceiver(acquisition, torch.zeros(2, dtype=torch.complex64),
        arm='recursive_pilot', phase_estimator='circular_clean')
    recorder = InputRecorder(KnownAdaptRealPhysicsBranch(model))
    decoder = ForwardCounter(recorder)
    logits, diagnostics = first.step(decoder, receiver_frame(frame))
    second.step(decoder, receiver_frame(frame))
    before_calls = decoder.calls
    state = {name: value.clone() for name, value in model.state_dict().items()}
    legal_state = (first.mean.clone(), first.covariance.clone(), first.tail.clone())
    scores = receiver_diagnostics(logits, frame.bits, frame.data_mask, diagnostics.pop('decode_cir'), frame.true_cir)
    scores.update(long_receiver_diagnostics(recorder.model, recorder.last_inputs, frame, phase, cfo, .1))
    poisoned = replace(frame, bits=~frame.bits, tx_symbols=-frame.tx_symbols, true_cir=frame.true_cir * 5)
    long_receiver_diagnostics(recorder.model, recorder.last_inputs, poisoned, phase + 1., cfo + .01, .1)
    assert decoder.calls == before_calls
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    for current, saved in zip((first.mean, first.covariance, first.tail), legal_state):
        torch.testing.assert_close(current, saved, rtol=0, atol=0)
    next_first, _ = first.step(decoder, receiver_frame(frame))
    next_second, _ = second.step(decoder, receiver_frame(frame))
    torch.testing.assert_close(next_first, next_second, rtol=0, atol=0)
    row = dict.fromkeys(STREAM_ROW_FIELDS, 1)
    row.update(diagnostics, **scores, true_phase=phase, data_labels=frame.bits)
    saved = json.loads(json.dumps(compact_stream_row(row, schema_version=3), allow_nan=False))
    for field in RECEIVER_SCORE_FIELDS + LONG_DIAGNOSTIC_FIELDS:
        assert saved[field] == scores[field]
    assert 'true_phase' not in saved and 'data_labels' not in saved


def test_schema3_rejects_missing_posthoc_scores_instead_of_silent_loss():
    row = dict.fromkeys(STREAM_ROW_FIELDS + RECEIVER_SCORE_FIELDS + LONG_DIAGNOSTIC_FIELDS, 1)
    del row['decoder_signal_mismatch_nmse_diagnostic']
    with pytest.raises(ValueError, match='缺少必要字段'):
        compact_stream_row(row, schema_version=3)
    assert 'aligned_nmse_diagnostic' not in compact_stream_row(row, schema_version=2)
