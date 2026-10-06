# -*- coding: utf-8 -*-

import pytest
import torch

from agent.pilot_state import PilotDriftDetector, PilotStateEmbedding, build_pilot_state_summary


def test_pilot_state_summary_and_embedding_are_bounded_and_label_free():
    summary = build_pilot_state_summary(
        cir=torch.tensor([1 + 0j, 0.1 + 0.0j]),
        reference_cir=torch.tensor([1 + 0j, 0.0 + 0.0j]),
        phase0=0.2,
        cfo_cycles_per_symbol=0.0006,
        noise_variance=0.3,
        confidence=0.8,
        reconstruction_error=0.4,
    )
    embedding = PilotStateEmbedding()(summary)
    assert embedding.shape == (6,)
    assert torch.all(embedding.abs() <= 1.0)
    assert summary.data_labels_used_online is False


def test_pilot_drift_detector_requires_confidence_and_change():
    detector = PilotDriftDetector(threshold=0.2, min_confidence=0.5)
    base = torch.zeros(6)
    changed = torch.ones(6)
    assert detector.update(base, confidence=1.0) == (False, 0.0)
    drift, distance = detector.update(changed, confidence=0.2)
    assert drift is False
    assert distance > 0.2
    drift, distance = detector.update(base, confidence=1.0)
    assert drift is True
    assert distance > 0.2


def test_invalid_state_shape_is_rejected():
    with pytest.raises(ValueError):
        PilotStateEmbedding(scales=torch.ones(5))
    with pytest.raises(ValueError):
        PilotDriftDetector(threshold=0.0)

