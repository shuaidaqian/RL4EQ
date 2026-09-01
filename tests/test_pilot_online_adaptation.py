from types import SimpleNamespace

import torch


def test_online_snr_layer_can_freeze_unreliable_peft_updates():
    import compare

    candidates = ({"name": "nominal", "groups": ("head",)},)
    assert compare._online_candidates_for_snr(candidates, snr_db=0.0, freeze_below_db=5.0) == ()
    assert compare._online_candidates_for_snr(candidates, snr_db=5.0, freeze_below_db=5.0) == candidates


def test_online_snr_layer_can_freeze_the_whole_update_chain():
    import compare

    assert compare._online_updates_are_frozen(0.0, 5.0) is True
    assert compare._online_updates_are_frozen(5.0, 5.0) is False
    assert compare._online_updates_are_frozen(10.0, None) is False

from agent.cir_estimator import condition_from_cir
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from training.online_adaptation import PilotDrivenOnlineAdapter, run_pilot_driven_online


def _identity_condition(batch: int = 1):
    cir = torch.zeros(batch, 5, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    return condition_from_cir(cir, snr_db=10.0)


def test_pilot_conditioner_depends_on_known_adapt_pilot():
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    rx_iq = torch.randn(1, 32, 2)
    adapt_mask = torch.zeros(1, 32, dtype=torch.bool)
    adapt_mask[:, :12] = True
    adapt_a = torch.zeros(1, 32, dtype=torch.complex64)
    adapt_a[:, :12] = 1.0 + 0.0j
    adapt_b = adapt_a.clone()
    adapt_b[:, 6] = -1.0 + 0.0j

    context_a = model._pilot_context(rx_iq, adapt_a, adapt_mask)
    context_b = model._pilot_context(rx_iq, adapt_b, adapt_mask)

    assert context_a.shape == (1, 24)
    assert not torch.allclose(context_a, context_b)


def test_online_adapter_updates_selected_peft_group_from_adapt_pilot_only():
    torch.manual_seed(5)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.where(
            torch.arange(32) % 2 == 0,
            torch.ones(32, dtype=torch.complex64),
            -torch.ones(32, dtype=torch.complex64),
        ),
        adapt_mask=torch.arange(32) < 12,
        reward_mask=torch.arange(32) >= 12,
        data_mask=torch.zeros(32, dtype=torch.bool),
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    adapter = PilotDrivenOnlineAdapter(model, groups={"head"}, learning_rate=1e-2, steps=1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    result = adapter.adapt(
        frame,
        _identity_condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
    )

    assert result.data_labels_used_online is False
    assert result.adapt_pilot_count == 12
    assert result.accepted is True
    assert result.parameter_delta_norm > 0.0
    changed = [
        name for name, value in model.named_parameters()
        if not torch.equal(before[name], value.detach())
    ]
    assert changed
    assert all(getattr(model.get_parameter(name), "_peft_group", None) == "head" for name in changed)


def test_online_adapter_rejects_frame_without_adapt_pilot():
    model = UnfoldedEqualizer(
        UnfoldedConfig(frame_len=16, max_delay=2, iterations=1, d_model=16, num_heads=4)
    )
    frame = SimpleNamespace(
        rx_symbols=torch.randn(16, dtype=torch.complex64),
        tx_symbols=torch.ones(16, dtype=torch.complex64),
        adapt_mask=torch.zeros(16, dtype=torch.bool),
        reward_mask=torch.ones(16, dtype=torch.bool),
        data_mask=torch.zeros(16, dtype=torch.bool),
        model_region_ids=torch.zeros(16, dtype=torch.long),
    )
    adapter = PilotDrivenOnlineAdapter(model, groups={"head"}, learning_rate=1e-2, steps=1)
    result = adapter.adapt(frame, _identity_condition(), torch.zeros(1, 2, dtype=torch.complex64))

    assert result.accepted is False
    assert result.adapt_pilot_count == 0
    assert result.parameter_delta_norm == 0.0


def test_online_adapter_accepts_action_specific_update_override():
    torch.manual_seed(8)
    model = UnfoldedEqualizer(
        UnfoldedConfig(
            frame_len=32,
            max_delay=4,
            iterations=1,
            d_model=24,
            num_heads=4,
            pilot_conditioned=True,
        )
    )
    frame = SimpleNamespace(
        rx_symbols=torch.randn(32, dtype=torch.complex64),
        tx_symbols=torch.where(
            torch.arange(32) % 2 == 0,
            torch.ones(32, dtype=torch.complex64),
            -torch.ones(32, dtype=torch.complex64),
        ),
        adapt_mask=torch.arange(32) < 12,
        reward_mask=torch.arange(32) >= 12,
        data_mask=torch.zeros(32, dtype=torch.bool),
        model_region_ids=torch.zeros(32, dtype=torch.long),
    )
    adapter = PilotDrivenOnlineAdapter(model, groups={"head"}, learning_rate=1e-2, steps=1)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}

    result = adapter.adapt(
        frame,
        _identity_condition(),
        torch.zeros(1, 4, dtype=torch.complex64),
        groups={"conditioner_film"},
        learning_rate=5e-3,
        steps=1,
        max_delta_norm=0.5,
    )

    assert result.accepted is True
    changed = [
        name for name, value in model.named_parameters()
        if not torch.equal(before[name], value.detach())
    ]
    assert changed
    assert all(
        getattr(model.get_parameter(name), "_peft_group", None) == "conditioner_film"
        for name in changed
    )


def test_pilot_online_runner_writes_online_only_metrics(tmp_path):
    result = run_pilot_driven_online(
        "configs/continual_ppo.json",
        frames=1,
        num_seeds=1,
        output_dir=tmp_path / "pilot_online",
        delays=[4],
        snrs=[10.0],
        pilot_total=64,
        pilot_layout="two_block",
        device="cpu",
    )

    assert result["schema_version"] == "pilot-driven-online-adaptation-v1"
    assert result["rows"][0]["online_update_source"] == "adapt_pilot_only"
    assert result["rows"][0]["data_labels_used_online"] is False


def test_pilot_online_runner_records_bandit_action_and_reward_boundary(tmp_path):
    result = run_pilot_driven_online(
        "configs/continual_ppo.json",
        frames=1,
        num_seeds=1,
        output_dir=tmp_path / "bandit_online",
        delays=[4],
        snrs=[10.0],
        pilot_total=64,
        pilot_layout="two_block",
        scheduler="bandit",
        device="cpu",
    )

    row = result["rows"][0]
    assert result["scheduler"] == "bandit"
    assert row["scheduler"] == "bandit"
    assert row["action"] in {"skip", "phase_weak", "head_weak", "head_nominal", "film_nominal", "joint_nominal"}
    assert row["uses_rl"] is True
    assert row["data_labels_used_online"] is False
    assert row["reward_pilot_loss_before"] >= 0.0
    assert row["reward_pilot_loss_after"] >= 0.0
