import torch

from baseline.traditional_equalizers import TRADITIONAL_BASELINES, run_traditional_equalizer
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from training.meta_training import _estimate_cir_from_known_frame


def test_traditional_baselines_are_distinct_non_neural_non_rl_algorithms():
    env = CommunicationEnvironment(CommEnvConfig(level="B", max_delay=20, snr_db=10.0, total_pilot=128, layout="multi_block", seed=7100))
    start = env.reset_episode()
    frame = env.next_frame()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)

    algorithms = set()
    for method in TRADITIONAL_BASELINES:
        result = run_traditional_equalizer(method, frame.receiver_view(), cir, start.initial_soft_tail, snr_db=10.0)
        algorithms.add(result.extra["algorithm"])
        assert result.logits.shape == frame.bits.shape
        assert result.extra["traditional"] is True
        assert result.extra["uses_neural_network"] is False
        assert result.extra["uses_rl"] is False
        assert result.extra["shared_placeholder_kernel"] is False

    assert algorithms == {
        "time_domain_lmmse_fir",
        "pilot_lms_linear",
        "pilot_nlms_linear",
        "pilot_rls_linear",
        "pilot_rls_dfe",
        "single_carrier_fde_mmse",
    }


def test_traditional_baselines_do_not_read_reward_or_data_labels():
    env = CommunicationEnvironment(CommEnvConfig(level="B", max_delay=20, snr_db=10.0, total_pilot=128, layout="multi_block", seed=7200))
    start = env.reset_episode()
    frame = env.next_frame()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)
    changed = frame.with_replaced_hidden_labels(
        reward_bits=torch.logical_not(frame.bits),
        data_bits=torch.logical_not(frame.bits),
    )

    for method in TRADITIONAL_BASELINES:
        result_a = run_traditional_equalizer(method, frame.receiver_view(), cir, start.initial_soft_tail, snr_db=10.0)
        result_b = run_traditional_equalizer(method, changed.receiver_view(), cir, start.initial_soft_tail, snr_db=10.0)
        assert torch.equal(result_a.logits, result_b.logits)
