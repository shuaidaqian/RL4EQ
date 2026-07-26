import torch
import subprocess
import sys

from agent.adaptation_controller import AdaptationController, compute_reward
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from agent.continual_policy import ContinualPolicy, ObservationEncoder, MODES, ITERATION_CHOICES
from agent.continual_policy import HierarchicalAction
from baseline.block_equalizers import DetectionResult
from training.continual_ppo import run_real_online_experiment, tiny_online_run, _initialize_safe_policy_prior
import training.continual_ppo as continual_ppo


class _HiddenLabelView:
    def __init__(self, reward_value: int = 0, data_value: int = 0):
        self.rx_symbols = torch.zeros(512, dtype=torch.complex64)
        self.adapt_symbols = torch.zeros(512, dtype=torch.complex64)
        self.adapt_mask = torch.zeros(512, dtype=torch.bool)
        self.adapt_mask[:96] = True
        self.model_region_ids = torch.zeros(512, dtype=torch.long)
        self.complex_cir = torch.zeros(41, dtype=torch.complex64)
        self.complex_cir[0] = 1.0 + 0.0j
        self.support_probability = torch.zeros(41)
        self.noise_variance = torch.tensor(0.01)
        self.confidence = torch.tensor(1.0)
        self.previous_reward = torch.tensor(0.2)
        self.last_parameter_delta_norm = torch.tensor(0.1)
        self.reward_bits = torch.full((512,), reward_value, dtype=torch.long)
        self.data_bits = torch.full((512,), data_value, dtype=torch.long)

    def with_hidden_labels(self, reward: int, data: int):
        return _HiddenLabelView(reward, data)


def test_policy_observation_excludes_reward_and_data_labels():
    view = _HiddenLabelView()
    obs_a = ObservationEncoder()(view.with_hidden_labels(reward=0, data=0))
    obs_b = ObservationEncoder()(view.with_hidden_labels(reward=1, data=1))
    assert torch.equal(obs_a.tensor, obs_b.tensor)
    assert "reward_bits" not in obs_a.fields
    assert "data_bits" not in obs_a.fields


def test_recurrent_policy_emits_legal_hierarchical_action():
    observation = ObservationEncoder()(_HiddenLabelView()).tensor.unsqueeze(0)
    policy = ContinualPolicy()
    action, log_prob, value, hidden = policy.sample(observation, torch.zeros(1, 1, 128))
    assert action.mode in {"skip", "update-channel", "update-equalizer", "joint-update", "detector-refine", "rollback"}
    assert action.steps in {1, 2, 4}
    assert action.detector_iterations in {2, 4, 6, 8}
    assert hidden.shape == (1, 1, 128)
    assert torch.isfinite(log_prob + value).all()


def test_policy_ablations_keep_contract_and_parameter_budget():
    observation = ObservationEncoder()(_HiddenLabelView()).tensor.unsqueeze(0)
    for ablation in ["none", "no_gru", "no_detector_control"]:
        policy = ContinualPolicy(ablation=ablation)
        action, log_prob, value, hidden = policy.sample(observation, torch.zeros(1, 1, 128))
        assert policy.parameter_count() < 1_000_000
        assert torch.isfinite(log_prob + value).all()
        if ablation == "no_detector_control":
            assert action.detector_iterations == 4
        assert hidden.shape == (1, 1, 128)


def test_safe_policy_prior_starts_from_best_fixed_like_action():
    policy = ContinualPolicy()
    _initialize_safe_policy_prior(policy)
    assert int(torch.argmax(policy.mode_head.bias).item()) == MODES.index("detector-refine")
    assert int(torch.argmax(policy.iter_head.bias).item()) == ITERATION_CHOICES.index(6)


def _controller_model():
    return UnfoldedEqualizer(UnfoldedConfig(frame_len=96, max_delay=4, iterations=1, d_model=24, num_heads=4, adapter_rank=4, lora_rank=4))


def test_controller_persists_safe_updates_and_resets_between_seeds():
    model = _controller_model()
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    controller = AdaptationController(model)
    action = HierarchicalAction("joint-update", "head", 1, 4, 1e-4, 1e-5, 0.0, 0.2, 1.0)

    controller.reset_episode(seed=1, checkpoint=checkpoint)
    before = controller.peft_vector().clone()
    result = controller.execute(action, reward_loss_before=torch.tensor(0.8), reward_loss_after=torch.tensor(0.7), shadow_loss=torch.tensor(0.9))
    after = controller.peft_vector().clone()

    assert result.reward > 0
    assert not torch.equal(before, after)
    assert torch.equal(after, controller.peft_vector())
    controller.reset_episode(seed=2, checkpoint=checkpoint)
    assert torch.equal(before, controller.peft_vector())


def test_nonfinite_update_hard_rolls_back():
    model = _controller_model()
    checkpoint = {name: value.detach().clone() for name, value in model.state_dict().items()}
    controller = AdaptationController(model)
    controller.reset_episode(seed=1, checkpoint=checkpoint)
    action = HierarchicalAction("update-equalizer", "head", 1, 4, float("nan"), 1e-5, 0.0, 0.2, 1.0)

    result = controller.execute(action, reward_loss_before=torch.tensor(0.8), reward_loss_after=torch.tensor(0.7), shadow_loss=torch.tensor(0.9))

    assert result.rollback
    assert result.reward == -1.0
    assert torch.equal(controller.peft_vector(), controller.last_safe_vector())


def test_shadow_reward_signature_excludes_data_labels():
    reward = compute_reward(torch.tensor(0.8), torch.tensor(0.4), torch.tensor(1.0), beta=0.5)
    assert reward > 0
    assert "data" not in compute_reward.__code__.co_varnames
    assert "ber" not in compute_reward.__code__.co_varnames


def test_continual_ppo_updates_during_deployment_without_cross_seed_state():
    run = tiny_online_run(frames=65, update_interval=32, seed=11)
    assert run.policy_update_frames == [32, 64]
    assert run.metrics[0].measured_before_current_frame_update
    second = tiny_online_run(frames=1, update_interval=32, seed=12)
    assert second.initial_policy_hash == run.offline_policy_hash
    assert second.initial_receiver_hash == run.offline_receiver_hash


def test_real_online_experiment_reports_level_b_frame_metrics(tmp_path):
    result = run_real_online_experiment(
        config_path="configs/continual_ppo.json",
        frames=2,
        num_seeds=1,
        update_interval=1,
        output_dir=tmp_path,
        delays=[20],
        snrs=[10],
    )
    assert result["schema_version"] == "continual-ppo-real-online-v1"
    assert result["policy_update_frames"] == [1, 2]
    assert len(result["rows"]) == 2
    assert all(row["level"] == "B" for row in result["rows"])
    assert all(row["method"] == "Continual PPO" for row in result["rows"])
    assert all(0.0 <= row["ber_data"] <= 1.0 for row in result["rows"])
    assert all(row["cir_update"] == "decision_directed" for row in result["rows"])
    assert result["policy_learning"] == "clipped_ppo_reward_pilot"
    assert any(row["policy_loss"] is not None for row in result["rows"])
    assert all("policy_action_mode" in row for row in result["rows"])
    assert (tmp_path / "online_metrics.json").exists()


def test_real_online_experiment_uses_receiver_soft_tail_after_first_frame(monkeypatch, tmp_path):
    captured_tails = []
    returned_tails = []

    def fake_detect(rx, cir, soft_tail, noise_variance, cg_iterations, refine_iterations):
        del cir, noise_variance, cg_iterations, refine_iterations
        captured_tails.append(soft_tail.detach().clone())
        logits = torch.zeros(rx.shape, dtype=torch.float32)
        marker = 0.25 + 0.25 * len(captured_tails)
        next_tail = torch.complex(
            torch.full_like(soft_tail.real, marker),
            torch.zeros_like(soft_tail.real),
        )
        returned_tails.append(next_tail.clone())
        return DetectionResult(
            logits=logits,
            probabilities=torch.full_like(logits, 0.5),
            soft_tail=next_tail,
            iterations=0,
        )

    monkeypatch.setattr(continual_ppo, "perfect_csi_bpsk_refine_detect", fake_detect)
    monkeypatch.setattr(continual_ppo, "_decision_directed_cir_update", lambda frame, logits, max_delay, previous_cir, alpha: previous_cir)

    run_real_online_experiment(
        config_path="configs/continual_ppo.json",
        frames=2,
        num_seeds=1,
        update_interval=1,
        output_dir=tmp_path,
        delays=[20],
        snrs=[10],
    )

    assert len(captured_tails) == 4
    assert torch.equal(captured_tails[2], returned_tails[1])


def test_online_train_cli_supports_config_slicing(tmp_path):
    subprocess.run(
        [
            sys.executable,
            "online_train.py",
            "--config",
            "configs/continual_ppo.json",
            "--frames",
            "1",
            "--num-seeds",
            "1",
            "--update-interval",
            "1",
            "--delays",
            "20",
            "--snrs",
            "10",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
    )
    assert (tmp_path / "online_metrics.json").exists()
