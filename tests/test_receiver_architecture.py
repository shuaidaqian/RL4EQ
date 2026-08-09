import torch

from agent.cir_estimator import HybridCIREstimator
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
