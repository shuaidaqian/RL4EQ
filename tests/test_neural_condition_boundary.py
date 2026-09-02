"""验证神经方法的在线信息边界，避免 Frozen 基线偷偷读取当前帧 Pilot。"""

import torch


def test_acquisition_condition_does_not_read_current_pilot(monkeypatch):
    import compare

    def fail_if_called(*args, **kwargs):
        raise AssertionError("严格 Frozen 条件不应读取当前帧 Pilot 相位特征")

    monkeypatch.setattr(compare, "estimate_phase_residual_vector", fail_if_called)
    features = compare._neural_phase_features(
        condition_source="acquisition",
        receiver_view=object(),
        cir=torch.ones(5, dtype=torch.complex64),
        soft_tail=torch.zeros(4, dtype=torch.complex64),
        acquisition_cfo=0.001,
    )

    assert features is None


def test_pilot_condition_reads_pilot_and_reports_source(monkeypatch):
    import compare

    expected = torch.arange(16, dtype=torch.float32)
    monkeypatch.setattr(compare, "estimate_phase_residual_vector", lambda *args, **kwargs: expected)
    features = compare._neural_phase_features(
        condition_source="pilot_phase",
        receiver_view=object(),
        cir=torch.ones(5, dtype=torch.complex64),
        soft_tail=torch.zeros(4, dtype=torch.complex64),
        acquisition_cfo=0.001,
    )

    assert torch.equal(features, expected)


def test_neural_method_audit_fields_distinguish_frozen_and_online():
    import compare

    assert compare._neural_method_contract("Offline NN only") == {
        "condition_source": "acquisition",
        "pilot_phase_used": False,
        "cir_update_applied": False,
        "peft_update_applied": False,
    }
    assert compare._neural_method_contract("Pilot-conditioned frozen NN") == {
        "condition_source": "pilot_phase",
        "pilot_phase_used": True,
        "cir_update_applied": False,
        "peft_update_applied": False,
    }
    assert compare._neural_method_contract("Pilot CIR only") == {
        "condition_source": "pilot_cir_phase",
        "pilot_phase_used": True,
        "cir_update_applied": True,
        "peft_update_applied": False,
    }


def test_diagnostic_method_group_exposes_phase_compensated_perfect_csi():
    from compare import method_group

    assert "Perfect-CSI + Pilot Phase" in method_group("diagnostic")
    assert "Perfect-CSI + Pilot Phase" not in method_group("traditional")
