"""验证 Pilot 驱动残差 Adapter 的 RLS 在线更新边界。"""

from types import SimpleNamespace

import pytest
import torch


def _condition(batch: int = 1):
    from agent.cir_estimator import condition_from_cir

    cir = torch.zeros(batch, 5, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    return condition_from_cir(cir, snr_db=10.0)


def _frame(frame_len: int = 32):
    tx = torch.where(
        torch.arange(frame_len) % 2 == 0,
        torch.ones(frame_len, dtype=torch.complex64),
        -torch.ones(frame_len, dtype=torch.complex64),
    )
    return SimpleNamespace(
        rx_symbols=torch.randn(frame_len, dtype=torch.complex64),
        tx_symbols=tx,
        bits=(tx.real > 0.0).float(),
        adapt_mask=torch.arange(frame_len) < 12,
        reward_mask=torch.arange(frame_len) >= 12,
        data_mask=torch.arange(frame_len) >= 16,
        model_region_ids=torch.zeros(frame_len, dtype=torch.long),
    )


def _model():
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    return UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )


def test_zero_initialized_residual_adapter_is_identical_to_frozen_model():
    from training.online_adaptation import PilotResidualRLSAdapter

    torch.manual_seed(21)
    model = _model()
    model.eval()
    frame = _frame()
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0)
    region_ids = frame.model_region_ids.unsqueeze(0)
    condition = _condition()
    tail = torch.zeros(1, 4, dtype=torch.complex64)

    with torch.no_grad():
        frozen, _ = model(rx_iq, condition, region_ids, tail)
    updater = PilotResidualRLSAdapter(model, forgetting_factor=0.99)
    with torch.no_grad():
        attached, _ = model(rx_iq, condition, region_ids, tail)

    assert torch.allclose(frozen, attached)
    assert updater.parameter_delta_norm == pytest.approx(0.0)


def test_residual_adapter_does_not_change_features_used_by_rls():
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    torch.manual_seed(24)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=2,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    model.attach_online_residual_adapter()
    frame = _frame()
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0)
    region_ids = frame.model_region_ids.unsqueeze(0)
    condition = _condition()
    tail = torch.zeros(1, 4, dtype=torch.complex64)

    with torch.no_grad():
        model(rx_iq, condition, region_ids, tail)
        features_before = model.online_residual_features.clone()
        model.online_residual_adapter.linear.bias.fill_(0.5)
        model(rx_iq, condition, region_ids, tail)
        features_after = model.online_residual_features.clone()

    assert torch.allclose(features_before, features_after)


def test_rls_updates_only_residual_adapter_from_effective_adapt_pilot():
    from training.online_adaptation import PilotResidualRLSAdapter

    torch.manual_seed(22)
    model = _model()
    before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }
    updater = PilotResidualRLSAdapter(
        model,
        forgetting_factor=0.99,
        warmup_symbols=4,
        target_logit=2.0,
        max_delta_norm=100.0,
    )

    result = updater.adapt(
        _frame(),
        _condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
    )

    assert result.accepted is True
    assert result.adapt_pilot_count == 8
    assert result.parameter_delta_norm > 0.0
    assert result.data_labels_used_online is False
    changed = [
        name for name, value in model.named_parameters()
        if not torch.equal(before.get(name, torch.empty(0)), value.detach())
    ]
    assert changed
    assert all(name.startswith("online_residual_adapter.") for name in changed)


def test_rls_rejects_frame_without_effective_adapt_pilot():
    from training.online_adaptation import PilotResidualRLSAdapter

    model = _model()
    updater = PilotResidualRLSAdapter(model, warmup_symbols=4)
    frame = _frame()
    frame.adapt_mask = torch.zeros_like(frame.adapt_mask)

    result = updater.adapt(
        frame,
        _condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
    )

    assert result.accepted is False
    assert result.adapt_pilot_count == 0
    assert result.parameter_delta_norm == pytest.approx(0.0)


def test_rls_enforces_total_adapter_drift_from_initial_anchor():
    from training.online_adaptation import PilotResidualRLSAdapter

    torch.manual_seed(25)
    model = _model()
    updater = PilotResidualRLSAdapter(
        model,
        warmup_symbols=4,
        max_delta_norm=100.0,
        max_total_delta_norm=1e-8,
    )

    result = updater.adapt(
        _frame(),
        _condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
    )

    assert result.accepted is False
    assert result.parameter_delta_norm == pytest.approx(0.0)
    assert result.cumulative_delta_norm == pytest.approx(0.0)
    assert result.max_total_delta_norm == pytest.approx(1e-8)
    assert torch.count_nonzero(updater.module.linear.weight) == 0
    assert torch.count_nonzero(updater.module.linear.bias) == 0


def test_rls_covariance_is_retained_across_frames():
    from training.online_adaptation import PilotResidualRLSAdapter

    torch.manual_seed(23)
    model = _model()
    updater = PilotResidualRLSAdapter(model, forgetting_factor=0.99)
    tail = torch.zeros(1, 4, dtype=torch.complex64)
    updater.adapt(_frame(), _condition(), tail)
    covariance_after_first = updater.covariance.detach().clone()
    updater.adapt(_frame(), _condition(), tail)

    assert not torch.allclose(updater.covariance, covariance_after_first)


def test_compare_can_build_the_rls_online_method_without_changing_checkpoint_shape():
    import compare

    config = {
        "online_adaptation_algorithm": "rls",
        "online_rls_forgetting_factor": 0.99,
        "online_rls_ridge": 1.0,
        "online_rls_warmup_symbols": 4,
        "online_rls_target_logit": 2.0,
        "online_rls_max_total_delta_norm": 0.5,
        "online_adaptation_max_delta_norm": 10.0,
        "online_condition_source": "acquisition",
        "online_rls_condition_source": "pilot_cir_phase",
        "model": {
            "frame_len": 32,
            "max_delay": 4,
            "iterations": 1,
            "d_model": 24,
            "num_heads": 4,
            "pilot_conditioned": True,
        },
    }
    states = compare._build_method_states(
        ("Pilot-Driven Online Adaptation",),
        torch.zeros(4, dtype=torch.complex64),
        torch.tensor([[1.0 + 0.0j, 0.0j, 0.0j, 0.0j, 0.0j]]),
        config,
        delay=4,
        snr_db=10.0,
        seed=0,
        device="cpu",
        online_condition_source_override="acquisition",
    )

    state = states["Pilot-Driven Online Adaptation"]
    assert isinstance(state, compare.PilotRLSMethodState)
    assert state.rls_adapter.parameter_delta_norm == pytest.approx(0.0)
    assert state.rls_adapter.max_total_delta_norm == pytest.approx(0.5)
    assert state.condition_source == "acquisition"
    assert state.model.online_residual_adapter is not None


def test_acquisition_condition_override_is_shared_by_frozen_and_rls_methods():
    import compare

    config = {
        "online_adaptation_algorithm": "rls",
        "online_rls_condition_source": "pilot_cir_phase",
        "model": {
            "frame_len": 32,
            "max_delay": 4,
            "iterations": 1,
            "d_model": 24,
            "num_heads": 4,
            "pilot_conditioned": True,
        },
    }
    states = compare._build_method_states(
        ("Frozen Offline NN", "Pilot-Driven Online Adaptation"),
        torch.zeros(4, dtype=torch.complex64),
        torch.tensor([[1.0 + 0.0j, 0.0j, 0.0j, 0.0j, 0.0j]]),
        config,
        delay=4,
        snr_db=10.0,
        seed=0,
        device="cpu",
        online_condition_source_override="acquisition",
    )

    assert states["Frozen Offline NN"].condition_source == "acquisition"
    assert states["Pilot-Driven Online Adaptation"].condition_source == "acquisition"
