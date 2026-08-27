import torch
import pytest

from baseline.traditional_equalizers import (
    TRADITIONAL_BASELINES,
    TraditionalPhaseState,
    estimate_phase_residual_features,
    estimate_phase_residual_vector,
    run_traditional_equalizer,
)
from baseline.synchronization_compensation import apply_phase_correction, estimate_pilot_phase_line
from baseline.traditional_equalizers import _estimate_cir_with_cfo_grid, _pilot_reference_from_cir
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from env.frame_structure import ReceiverFrameView
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
        "cfo_corrected_time_domain_lmmse_fir",
        "cfo_corrected_pilot_rls_dfe",
        "dd_phase_tracked_time_domain_lmmse_fir",
        "dd_phase_tracked_pilot_rls_dfe",
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


def test_pilot_based_cfo_correction_reduces_known_rotation_on_pilots():
    frame_len = 128
    indices = torch.arange(frame_len, dtype=torch.float32)
    tx = torch.where((indices.long() % 2) == 0, 1.0, -1.0).to(torch.complex64)
    phase0 = 0.4
    cfo = 0.006
    rx = tx * torch.exp(1j * (phase0 + 2.0 * torch.pi * cfo * indices))
    adapt_mask = torch.zeros(frame_len, dtype=torch.bool)
    adapt_mask[:64] = True
    adapt_symbols = torch.zeros_like(tx)
    adapt_symbols[adapt_mask] = tx[adapt_mask]
    view = ReceiverFrameView(
        rx_symbols=rx,
        adapt_symbols=adapt_symbols,
        adapt_mask=adapt_mask,
        model_region_ids=torch.zeros(frame_len, dtype=torch.long),
    )

    estimate = estimate_pilot_phase_line(view)
    corrected = apply_phase_correction(rx, estimate.phase0, estimate.cfo_cycles_per_symbol)
    before_error = torch.mean(torch.abs(torch.angle(rx[adapt_mask] * torch.conj(tx[adapt_mask]))))
    after_error = torch.mean(torch.abs(torch.angle(corrected[adapt_mask] * torch.conj(tx[adapt_mask]))))

    assert estimate.cfo_cycles_per_symbol == pytest.approx(cfo, abs=1e-7)
    assert after_error < before_error * 0.05


def test_compensated_traditional_baselines_are_non_neural_non_rl_and_label_free():
    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=20,
            snr_db=10.0,
            total_pilot=128,
            layout="prefix",
            seed=8400,
            impairment_profile="cfo_phase_light",
        )
    )
    start = env.reset_episode()
    frame = env.next_frame()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)
    changed = frame.with_replaced_hidden_labels(
        reward_bits=torch.logical_not(frame.bits),
        data_bits=torch.logical_not(frame.bits),
    )

    for method in ["CFO-Corrected DFE-RLS", "CFO-Corrected LMMSE-FIR"]:
        result_a = run_traditional_equalizer(method, frame.receiver_view(), cir, start.initial_soft_tail, snr_db=10.0)
        result_b = run_traditional_equalizer(method, changed.receiver_view(), cir, start.initial_soft_tail, snr_db=10.0)
        assert torch.equal(result_a.logits, result_b.logits)
        assert result_a.extra["traditional"] is True
        assert result_a.extra["uses_neural_network"] is False
        assert result_a.extra["uses_rl"] is False
        assert result_a.extra["uses_reward_or_data_labels"] is False
        assert result_a.extra["phase_compensation"] == "pilot_common_phase"


def test_dd_phase_tracked_traditional_baselines_are_label_free():
    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=20,
            snr_db=10.0,
            total_pilot=128,
            layout="prefix",
            seed=8410,
            impairment_profile="cfo_phase_light",
        )
    )
    start = env.reset_episode()
    frame = env.next_frame()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)
    changed = frame.with_replaced_hidden_labels(
        reward_bits=torch.logical_not(frame.bits),
        data_bits=torch.logical_not(frame.bits),
    )

    for method in ["CFO+DD-Phase DFE-RLS", "CFO+DD-Phase LMMSE-FIR"]:
        state_a = TraditionalPhaseState()
        state_b = TraditionalPhaseState()
        result_a = run_traditional_equalizer(method, frame.receiver_view(), cir, start.initial_soft_tail, snr_db=10.0, phase_state=state_a)
        result_b = run_traditional_equalizer(method, changed.receiver_view(), cir, start.initial_soft_tail, snr_db=10.0, phase_state=state_b)
        assert torch.equal(result_a.logits, result_b.logits)
        assert result_a.extra["traditional"] is True
        assert result_a.extra["uses_neural_network"] is False
        assert result_a.extra["uses_rl"] is False
        assert result_a.extra["uses_reward_or_data_labels"] is False
        assert result_a.extra["phase_compensation"] == "pilot_linear_fit+decision_directed_tracker"


def test_phase_residual_features_are_label_free_and_nonzero_under_rotation():
    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=20,
            snr_db=20.0,
            total_pilot=128,
            layout="prefix",
            seed=8430,
            cfo_cycles_per_symbol=0.0005,
            phase_noise_std=0.0002,
        )
    )
    start = env.reset_episode()
    frame = env.next_frame()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)
    changed = frame.with_replaced_hidden_labels(
        reward_bits=torch.logical_not(frame.bits),
        data_bits=torch.logical_not(frame.bits),
    )

    phase_a, cfo_a = estimate_phase_residual_features(frame.receiver_view(), cir, start.initial_soft_tail)
    phase_b, cfo_b = estimate_phase_residual_features(changed.receiver_view(), cir, start.initial_soft_tail)

    assert phase_a == pytest.approx(phase_b)
    assert cfo_a == pytest.approx(cfo_b)
    assert abs(phase_a) > 1e-5 or abs(cfo_a) > 1e-6


def test_phase_residual_vector_splits_adapt_pilot_into_local_statistics():
    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=20,
            snr_db=20.0,
            total_pilot=128,
            layout="prefix",
            seed=8435,
            cfo_cycles_per_symbol=0.0005,
            phase_noise_std=0.0002,
        )
    )
    start = env.reset_episode()
    frame = env.next_frame()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)
    changed = frame.with_replaced_hidden_labels(
        reward_bits=torch.logical_not(frame.bits),
        data_bits=torch.logical_not(frame.bits),
    )

    vector_a = estimate_phase_residual_vector(frame.receiver_view(), cir, start.initial_soft_tail, blocks=4)
    vector_b = estimate_phase_residual_vector(changed.receiver_view(), cir, start.initial_soft_tail, blocks=4)

    assert vector_a.shape == (16,)
    assert torch.allclose(vector_a, vector_b)
    assert torch.count_nonzero(vector_a).item() >= 4


def test_pilot_phase_line_is_bounded_under_cfo_phase_tiny_long_isi():
    """长 ISI 下逐帧 Pilot 相位拟合不能把 tiny CFO 放大到 1e-2 量级。"""

    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=20,
            snr_db=10.0,
            total_pilot=128,
            layout="prefix",
            seed=50002,
            impairment_profile="cfo_phase_tiny",
        )
    )
    start = env.reset_episode()
    frame = env.next_frame()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)
    reference, reliable = _pilot_reference_from_cir(frame.receiver_view(), cir, start.initial_soft_tail)

    estimate = estimate_pilot_phase_line(
        frame.receiver_view(),
        reference_symbols=reference,
        reference_mask=reliable,
        max_abs_cfo_cycles_per_symbol=0.00399,
    )

    assert int(reliable.sum().item()) >= 32
    assert abs(estimate.cfo_cycles_per_symbol) <= 0.004


def test_cfo_corrected_lmmse_does_not_overrotate_cfo_phase_tiny_smoke():
    """补偿版传统 baseline 不能因过估计 CFO 而在 tiny profile 中显著劣于未补偿版本。"""

    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=20,
            snr_db=10.0,
            total_pilot=128,
            layout="prefix",
            seed=50002,
            impairment_profile="cfo_phase_tiny",
        )
    )
    start = env.reset_episode()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)
    plain_tail = start.initial_soft_tail.clone()
    corrected_tail = start.initial_soft_tail.clone()
    plain_bers = []
    corrected_bers = []
    estimated_cfos = []
    for _ in range(3):
        frame = env.next_frame()
        plain = run_traditional_equalizer("LMMSE-FIR", frame.receiver_view(), cir, plain_tail, snr_db=10.0)
        corrected = run_traditional_equalizer("CFO-Corrected LMMSE-FIR", frame.receiver_view(), cir, corrected_tail, snr_db=10.0)
        plain_tail = plain.soft_tail
        corrected_tail = corrected.soft_tail
        plain_bers.append(torch.mean(((plain.logits[frame.data_mask] > 0) != frame.bits[frame.data_mask]).float()).item())
        corrected_bers.append(torch.mean(((corrected.logits[frame.data_mask] > 0) != frame.bits[frame.data_mask]).float()).item())
        estimated_cfos.append(abs(float(corrected.extra["estimated_cfo_cycles_per_symbol"])))

    assert max(estimated_cfos) <= 0.004
    assert sum(corrected_bers) / len(corrected_bers) <= sum(plain_bers) / len(plain_bers) + 0.05


def test_dd_phase_tracker_does_not_degrade_multiframe_cfo_phase_smoke():
    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=20,
            snr_db=10.0,
            total_pilot=128,
            layout="prefix",
            seed=8420,
            impairment_profile="cfo_phase_light",
        )
    )
    start = env.reset_episode()
    from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo

    cir, _ = estimate_acquisition_cir_with_cfo(start.acquisition, 20)
    plain_tail = start.initial_soft_tail.clone()
    tracked_tail = start.initial_soft_tail.clone()
    phase_state = TraditionalPhaseState()
    plain_bers = []
    tracked_bers = []
    for _ in range(4):
        frame = env.next_frame()
        plain = run_traditional_equalizer("CFO-Corrected LMMSE-FIR", frame.receiver_view(), cir, plain_tail, snr_db=10.0)
        tracked = run_traditional_equalizer(
            "CFO+DD-Phase LMMSE-FIR",
            frame.receiver_view(),
            cir,
            tracked_tail,
            snr_db=10.0,
            phase_state=phase_state,
        )
        plain_tail = plain.soft_tail
        tracked_tail = tracked.soft_tail
        plain_bers.append(torch.mean(((plain.logits[frame.data_mask] > 0) != frame.bits[frame.data_mask]).float()).item())
        tracked_bers.append(torch.mean(((tracked.logits[frame.data_mask] > 0) != frame.bits[frame.data_mask]).float()).item())

    assert sum(tracked_bers) / len(tracked_bers) <= sum(plain_bers) / len(plain_bers) + 1e-6
    assert abs(phase_state.phase0) > 0.0 or abs(phase_state.cfo_cycles_per_symbol) > 0.0


def test_traditional_baseline_updates_soft_tail_between_frames():
    env = CommunicationEnvironment(
        CommEnvConfig(level="B", max_delay=20, snr_db=10.0, total_pilot=128, layout="prefix", seed=8450)
    )
    start = env.reset_episode()
    frame = env.next_frame()
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)

    result = run_traditional_equalizer("LMMSE-FIR", frame.receiver_view(), cir, start.initial_soft_tail, snr_db=10.0)

    assert result.soft_tail.shape == start.initial_soft_tail.shape
    assert not torch.equal(result.soft_tail, start.initial_soft_tail)


def test_cfo_correction_uses_cir_reference_under_isi():
    """长 ISI 下不能直接把 rx[pilot] 当作当前 pilot；必须先构造 CIR 参考波形。"""

    frame_len = 96
    max_delay = 4
    indices = torch.arange(frame_len, dtype=torch.float32)
    bits = torch.tensor([(index * 7) % 11 > 4 for index in range(frame_len)])
    tx = torch.complex(bits.to(torch.float32) * 2.0 - 1.0, torch.zeros(frame_len))
    adapt_mask = torch.zeros(frame_len, dtype=torch.bool)
    adapt_mask[:64] = True
    adapt_symbols = torch.zeros_like(tx)
    adapt_symbols[adapt_mask] = tx[adapt_mask]
    cir = torch.zeros(max_delay + 1, dtype=torch.complex64)
    cir[0] = 0.75 + 0.0j
    cir[4] = 0.55 + 0.25j
    cir = cir / torch.linalg.norm(cir)
    tail = torch.ones(max_delay, dtype=torch.complex64)
    padded = torch.cat((tail, tx))
    clean = torch.zeros(frame_len, dtype=torch.complex64)
    for delay, tap in enumerate(cir):
        if tap.abs() > 0:
            clean += tap * padded[max_delay + torch.arange(frame_len) - delay]
    phase0 = -0.7
    cfo = 0.004
    rx = clean * torch.exp(1j * (phase0 + 2.0 * torch.pi * cfo * indices))
    view = ReceiverFrameView(
        rx_symbols=rx,
        adapt_symbols=adapt_symbols,
        adapt_mask=adapt_mask,
        model_region_ids=torch.zeros(frame_len, dtype=torch.long),
    )

    reference, reliable = _pilot_reference_from_cir(view, cir, tail)
    estimate = estimate_pilot_phase_line(view, reference_symbols=reference, reference_mask=reliable)

    assert int(reliable.sum().item()) >= 32
    assert estimate.cfo_cycles_per_symbol == pytest.approx(cfo, abs=2e-4)


def test_acquisition_cir_grid_search_estimates_cfo_without_true_impairment():
    frame_len = 512
    max_delay = 6
    generator = torch.Generator(device="cpu").manual_seed(8500)
    bits = torch.randint(0, 2, (frame_len,), generator=generator, dtype=torch.int64).bool()
    tx = torch.complex(bits.to(torch.float32) * 2.0 - 1.0, torch.zeros(frame_len))
    cir = torch.zeros(max_delay + 1, dtype=torch.complex64)
    cir[0] = 0.8 + 0.0j
    cir[3] = 0.25 - 0.15j
    cir[6] = -0.35 + 0.2j
    cir = cir / torch.linalg.norm(cir)
    clean = torch.zeros(frame_len, dtype=torch.complex64)
    for delay, tap in enumerate(cir):
        clean[delay:] += tap * tx[: frame_len - delay]
    cfo = -0.0025
    phase0 = 0.35
    idx = torch.arange(frame_len, dtype=torch.float32)
    rx = clean * torch.exp(1j * (phase0 + 2.0 * torch.pi * cfo * idx))

    estimated_cir, estimated_cfo = _estimate_cir_with_cfo_grid(tx, rx, max_delay, cfo_limit=0.004, grid_points=41)

    assert estimated_cfo == pytest.approx(cfo, abs=2.5e-4)
    nmse = torch.sum(torch.abs(estimated_cir - cir * torch.exp(1j * torch.angle(estimated_cir[0] / cir[0]))) ** 2)
    assert float(nmse.item()) < 0.2
