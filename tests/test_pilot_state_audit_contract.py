# -*- coding: utf-8 -*-

def test_online_state_audit_contract_is_label_free():
    from agent.pilot_state import PilotDriftDetector, PilotStateEmbedding, build_pilot_state_summary
    import torch

    summary = build_pilot_state_summary(
        cir=torch.ones(4, dtype=torch.complex64),
        reference_cir=torch.ones(4, dtype=torch.complex64),
        confidence=1.0,
    )
    embedding = PilotStateEmbedding()(summary)
    detector = PilotDriftDetector()
    drift, distance = detector.update(embedding, summary.confidence)
    assert embedding.numel() == 6
    assert drift is False
    assert distance == 0.0
    assert summary.data_labels_used_online is False
