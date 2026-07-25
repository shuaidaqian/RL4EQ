import torch

from agent.continual_policy import ContinualPolicy, ObservationEncoder


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
