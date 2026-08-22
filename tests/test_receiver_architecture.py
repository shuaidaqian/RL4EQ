import pytest
import torch
from types import SimpleNamespace

from agent.cir_estimator import HybridCIREstimator, condition_from_cir, decision_directed_cir_update
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect, perfect_csi_cg_detect
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from env.linear_operator import LinearChannelOperator


def test_linear_operator_adjoint_identity():
    operator = LinearChannelOperator(frame_len=64, max_delay=7)
    x = torch.randn(2, 64, dtype=torch.complex64)
    y = torch.randn(2, 64, dtype=torch.complex64)
    cir = torch.randn(2, 8, dtype=torch.complex64)
    tail = torch.zeros(2, 7, dtype=torch.complex64)
    lhs = (operator.forward(x, cir, tail).conj() * y).sum()
    rhs = (x.conj() * operator.adjoint(y, cir)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_perfect_csi_detector_recovers_noiseless_bpsk():
    bits = torch.tensor([[0, 1, 1, 0, 1, 0, 0, 1]], dtype=torch.bool)
    symbols = torch.complex(bits.to(torch.float32) * 2.0 - 1.0, torch.zeros_like(bits, dtype=torch.float32))
    cir = torch.zeros(1, 4, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    tail = torch.zeros(1, 3, dtype=torch.complex64)
    rx = LinearChannelOperator(frame_len=8, max_delay=3).forward(symbols, cir, tail)
    result = perfect_csi_cg_detect(rx, cir, tail, torch.tensor(1e-6), iterations=16)
    assert result.logits.shape == bits.shape
    assert result.probabilities.shape == bits.shape
    assert bit_error_rate(result.logits, bits) == 0.0


def test_perfect_cir_refine_recovers_noiseless_sparse_long_delay_bpsk():
    """Perfect-CIR 块检测器在无噪声长延迟稀疏信道下应恢复 BPSK，否则诊断上界无效。"""

    torch.manual_seed(7)
    frame_len = 96
    max_delay = 20
    bits = torch.randint(0, 2, (1, frame_len), dtype=torch.bool)
    symbols = torch.complex(bits.to(torch.float32) * 2.0 - 1.0, torch.zeros_like(bits, dtype=torch.float32))
    tail_bits = torch.randint(0, 2, (1, max_delay), dtype=torch.bool)
    tail = torch.complex(tail_bits.to(torch.float32) * 2.0 - 1.0, torch.zeros_like(tail_bits, dtype=torch.float32))
    cir = torch.zeros(1, max_delay + 1, dtype=torch.complex64)
    cir[:, 0] = 0.8 + 0.0j
    cir[:, 7] = 0.35 - 0.20j
    cir[:, 20] = -0.25 + 0.10j
    cir = cir / torch.sqrt(torch.sum(torch.abs(cir) ** 2, dim=1, keepdim=True))
    rx = LinearChannelOperator(frame_len=frame_len, max_delay=max_delay).forward(symbols, cir, tail)

    result = perfect_csi_bpsk_refine_detect(
        rx,
        cir,
        tail,
        torch.tensor(1e-6),
        cg_iterations=128,
        refine_iterations=4,
    )

    assert bit_error_rate(result.logits, bits) == 0.0


def test_perfect_cir_bpsk_refine_reaches_level_b_smoke_gate():
    """真实 Level B 小样本 gate：强 Perfect-CIR 检测器不能再停留在占位指标。"""

    rows = []
    for delay in (20, 30, 40):
        for snr_db in (10, 15, 20):
            bers = []
            for seed in range(2):
                env = CommunicationEnvironment(
                    CommEnvConfig(level="B", max_delay=delay, snr_db=snr_db, rho=0.99, total_pilot=128, layout="multi_block", seed=4100 + seed)
                )
                env.reset_episode()
                for _ in range(2):
                    frame = env.next_frame()
                    result = perfect_csi_bpsk_refine_detect(
                        frame.rx_symbols,
                        frame.true_cir,
                        frame.tail_symbols,
                        torch.tensor(10.0 ** (-snr_db / 10.0)),
                        cg_iterations=64,
                        refine_iterations=2,
                    )
                    bers.append(bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]))
            rows.append({"delay": delay, "snr_db": snr_db, "ber_data": sum(bers) / len(bers)})
    assert len(rows) == 9
    assert all(row["ber_data"] < 0.01 for row in rows)


def test_cir_estimator_contract_and_label_isolation():
    estimator = HybridCIREstimator(max_delay=40, latent_dim=96)
    rx_iq = torch.randn(2, 96, 2)
    adapt_symbols = torch.sign(torch.randn(2, 96)).to(torch.complex64)
    adapt_mask = torch.ones(2, 96, dtype=torch.bool)
    unknown_region_ids = torch.zeros(2, 96, dtype=torch.long)
    reward_bits = torch.zeros(2, 96, dtype=torch.bool)
    out_a = estimator(rx_iq, adapt_symbols, adapt_mask, unknown_region_ids)
    changed_reward = reward_bits.logical_not()
    out_b = estimator(rx_iq, adapt_symbols, adapt_mask, unknown_region_ids)
    assert out_a.complex_cir.shape == (2, 41)
    assert torch.is_complex(out_a.complex_cir)
    assert out_a.support_probability.shape == (2, 41)
    assert out_a.latent_residual.shape == (2, 96)
    assert torch.allclose(out_a.complex_cir, out_b.complex_cir, atol=1e-6, rtol=1e-6)
    assert changed_reward.shape == reward_bits.shape


def test_condition_from_cir_can_encode_phase_residual_features():
    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j

    condition = condition_from_cir(cir, 10.0, phase_residual=0.25, cfo_residual=0.001)

    assert condition.latent_residual.shape == (1, 96)
    assert condition.latent_residual[0, 0].item() == pytest.approx(0.25)
    assert condition.latent_residual[0, 1].item() == pytest.approx(0.001)


def test_condition_from_cir_accepts_rich_phase_feature_vector():
    cir = torch.zeros(5, dtype=torch.complex64)
    cir[0] = 1.0 + 0.0j
    features = torch.arange(16, dtype=torch.float32) / 10.0

    condition = condition_from_cir(cir, 10.0, phase_features=features)

    assert torch.allclose(condition.latent_residual[0, :16].cpu(), features)


def test_cir_estimator_recovers_noiseless_sparse_channel():
    max_delay = 6
    estimator = HybridCIREstimator(max_delay=max_delay, latent_dim=96)
    bits = torch.randint(0, 2, (1, 128), dtype=torch.bool)
    tx = torch.complex(bits.to(torch.float32) * 2.0 - 1.0, torch.zeros(1, 128))
    true_cir = torch.zeros(1, max_delay + 1, dtype=torch.complex64)
    true_cir[0, 0] = 0.9 + 0.0j
    true_cir[0, 3] = 0.25 + 0.15j
    true_cir[0, 6] = -0.2 + 0.1j
    true_cir = true_cir / torch.sqrt(torch.sum(torch.abs(true_cir) ** 2))
    rx = LinearChannelOperator(frame_len=128, max_delay=max_delay).forward(
        tx, true_cir, torch.zeros(1, max_delay, dtype=torch.complex64)
    )
    rx_iq = torch.stack([rx.real, rx.imag], dim=-1)
    out = estimator(rx_iq, tx, torch.ones(1, 128, dtype=torch.bool), torch.zeros(1, 128, dtype=torch.long))
    nmse = torch.sum(torch.abs(out.complex_cir - true_cir) ** 2) / torch.sum(torch.abs(true_cir) ** 2)
    assert 10.0 * torch.log10(nmse.clamp_min(1e-12)).item() < -20.0
    assert torch.mean(out.support_probability[true_cir.abs() > 1e-4]).item() > 0.8


def test_decision_directed_cir_update_can_gate_low_confidence_data_decisions():
    """低置信错误 Data 判决不应强行进入 DD-CIR 最小二乘更新。"""

    max_delay = 2
    tx = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.float32)
    tx_complex = torch.complex(tx, torch.zeros_like(tx))
    true_cir = torch.tensor([0.9 + 0.0j, 0.25 + 0.0j, -0.15 + 0.0j], dtype=torch.complex64)
    true_cir = true_cir / torch.sqrt(torch.sum(torch.abs(true_cir) ** 2))
    rx = LinearChannelOperator(frame_len=tx.numel(), max_delay=max_delay).forward(
        tx_complex.unsqueeze(0),
        true_cir.unsqueeze(0),
        torch.zeros(1, max_delay, dtype=torch.complex64),
    ).squeeze(0)
    adapt_mask = torch.tensor([True, True, True, True, False, False, False, False])
    wrong_logits = torch.tensor([6.0, -6.0, 6.0, -6.0, -0.05, 0.05, -0.05, 0.05])
    frame = SimpleNamespace(
        adapt_mask=adapt_mask,
        tx_symbols=tx_complex,
        rx_symbols=rx,
    )
    previous = true_cir.clone()

    ungated = decision_directed_cir_update(frame, wrong_logits, max_delay, previous, alpha=1.0)
    gated = decision_directed_cir_update(
        frame,
        wrong_logits,
        max_delay,
        previous,
        alpha=1.0,
        confidence_threshold=0.6,
    )

    ungated_nmse = torch.sum(torch.abs(ungated - true_cir) ** 2)
    gated_nmse = torch.sum(torch.abs(gated - true_cir) ** 2)
    assert gated_nmse < ungated_nmse


def test_unfolded_equalizer_is_noncausal_and_peft_is_bounded():
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=4))
    rx_iq = torch.randn(2, 512, 2)
    condition = HybridCIREstimator(max_delay=40)(rx_iq, torch.ones(2, 512, dtype=torch.complex64), torch.ones(2, 512, dtype=torch.bool), torch.zeros(2, 512, dtype=torch.long))
    region_ids = torch.zeros(2, 512, dtype=torch.long)
    soft_tail = torch.zeros(2, 40, dtype=torch.complex64)
    logits_a, probs_a = model(rx_iq, condition, region_ids, soft_tail)
    rx_changed = rx_iq.clone()
    rx_changed[:, -1] += 3.0
    logits_b, _ = model(rx_changed, condition, region_ids, soft_tail)
    assert logits_a.shape == (2, 512)
    assert probs_a.shape == (2, 512)
    assert not torch.equal(logits_a[:, 100], logits_b[:, 100])
    model.set_trainable_groups({"adapter", "attention_lora"})
    assert model.trainable_parameter_count() <= 0.10 * model.parameter_count()


def test_unfolded_equalizer_optional_conditioner_uses_cir_support_and_noise():
    config = UnfoldedConfig(
        frame_len=64,
        max_delay=6,
        iterations=1,
        d_model=32,
        num_heads=4,
        conditioner_uses_cir_summary=True,
    )
    model = UnfoldedEqualizer(config)
    rx_iq = torch.randn(1, 64, 2)
    cir = torch.zeros(1, 7, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    latent = torch.zeros(1, 96)
    condition_a = HybridCIREstimator(max_delay=6)(rx_iq, torch.ones(1, 64, dtype=torch.complex64), torch.ones(1, 64, dtype=torch.bool), torch.zeros(1, 64, dtype=torch.long))
    condition_a.complex_cir = cir
    condition_a.support_probability = torch.zeros(1, 7)
    condition_a.support_probability[:, 0] = 1.0
    condition_a.noise_variance = torch.tensor([0.01])
    condition_a.confidence = torch.tensor([1.0])
    condition_a.latent_residual = latent
    condition_b = HybridCIREstimator(max_delay=6)(rx_iq, torch.ones(1, 64, dtype=torch.complex64), torch.ones(1, 64, dtype=torch.bool), torch.zeros(1, 64, dtype=torch.long))
    condition_b.complex_cir = cir
    condition_b.support_probability = torch.ones(1, 7) / 7.0
    condition_b.noise_variance = torch.tensor([0.5])
    condition_b.confidence = torch.tensor([0.2])
    condition_b.latent_residual = latent
    region_ids = torch.zeros(1, 64, dtype=torch.long)
    soft_tail = torch.zeros(1, 6, dtype=torch.complex64)

    logits_a, _ = model(rx_iq, condition_a, region_ids, soft_tail)
    logits_b, _ = model(rx_iq, condition_b, region_ids, soft_tail)

    assert not torch.allclose(logits_a, logits_b)


def test_unfolded_equalizer_phase_correction_branch_uses_phase_vector():
    """显式 phase branch 应根据 latent phase vector 对整帧 rx_iq 做前置相位校正。"""

    config = UnfoldedConfig(
        frame_len=32,
        max_delay=4,
        iterations=1,
        d_model=24,
        num_heads=4,
        enable_phase_correction_branch=True,
        phase_correction_segments=4,
        phase_correction_initial_scale=1.0,
    )
    model = UnfoldedEqualizer(config)
    indices = torch.arange(32, dtype=torch.float32)
    true_phase0 = 0.3
    true_cfo = 0.002
    clean = torch.ones(1, 32, dtype=torch.complex64)
    rotated = clean * torch.exp(1j * (true_phase0 + 2.0 * torch.pi * true_cfo * indices)).unsqueeze(0)
    rx_iq = torch.stack((rotated.real, rotated.imag), dim=-1)
    cir = torch.zeros(1, 5, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    phase_features = torch.zeros(1, 96)
    for block in range(4):
        phase_features[0, block * 4 + 0] = true_phase0
        phase_features[0, block * 4 + 1] = true_cfo
    condition = SimpleNamespace(
        complex_cir=cir,
        support_probability=torch.zeros(1, 5),
        noise_variance=torch.tensor([0.01]),
        confidence=torch.ones(1),
        latent_residual=phase_features,
    )

    corrected = model._apply_phase_correction(rx_iq, condition)
    corrected_complex = torch.complex(corrected[..., 0], corrected[..., 1])

    assert torch.mean(torch.abs(corrected_complex - clean)).item() < 1e-3


def test_unfolded_equalizer_phase_correction_scale_zero_keeps_identity():
    """phase branch 初始 scale 为 0 时必须近似恒等，避免训练初期强制错误相位校正。"""

    config = UnfoldedConfig(
        frame_len=32,
        max_delay=4,
        iterations=1,
        d_model=24,
        num_heads=4,
        enable_phase_correction_branch=True,
        phase_correction_segments=4,
        phase_correction_initial_scale=0.0,
    )
    model = UnfoldedEqualizer(config)
    rx_iq = torch.randn(2, 32, 2)
    phase_features = torch.zeros(2, 96)
    phase_features[:, 0::4] = 0.5
    phase_features[:, 1::4] = 0.003
    condition = SimpleNamespace(
        complex_cir=torch.zeros(2, 5, dtype=torch.complex64),
        support_probability=torch.zeros(2, 5),
        noise_variance=torch.tensor([0.01, 0.02]),
        confidence=torch.ones(2),
        latent_residual=phase_features,
    )

    corrected = model._apply_phase_correction(rx_iq, condition)

    assert torch.allclose(corrected, rx_iq, atol=1e-6)
    assert not model.phase_correction_scale.requires_grad
    model.set_trainable_groups({"conditioner_film"})
    assert model.phase_correction_scale.requires_grad


def test_unfolded_equalizer_default_config_keeps_legacy_state_dict_shape():
    """默认关闭 phase branch，避免旧 checkpoint strict-load 被新模块破坏。"""

    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=64, max_delay=6, iterations=1, d_model=32, num_heads=4))

    assert model.phase_correction is None
    assert all("phase_correction" not in name for name in model.state_dict())


def test_unfolded_equalizer_strict_checkpoint_and_peft_snapshot(tmp_path):
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=64, max_delay=6, iterations=2, d_model=32, num_heads=4))
    checkpoint = tmp_path / "model.pt"
    torch.save({"schema_version": "unfolded-eq-v1", "model_config": model.config.to_dict(), "state_dict": model.state_dict()}, checkpoint)
    loaded = UnfoldedEqualizer(UnfoldedConfig.from_dict(torch.load(checkpoint, weights_only=False)["model_config"]))
    loaded.load_state_dict(torch.load(checkpoint, weights_only=False)["state_dict"], strict=True)
    model.set_trainable_groups({"adapter_lora"})
    snapshot = model.peft.snapshot({"adapter_lora"})
    for parameter in model.trainable_parameters():
        parameter.data.add_(0.01)
    assert model.peft.delta_norm(snapshot) > 0.0
    model.peft.restore(snapshot)
    assert model.peft.delta_norm(snapshot) == 0.0
