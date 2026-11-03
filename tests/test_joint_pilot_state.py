# -*- coding: utf-8 -*-
"""共同 Pilot 状态的坐标、后验、过程模型与隐藏标签隔离。"""

from dataclasses import replace

import pytest
import torch

from evaluation.joint_pilot_state import (
    PilotBlock, extract_adapt_blocks, joint_posterior, recursive_posterior, transport_block,
)


def _problem():
    rng = torch.Generator().manual_seed(834)
    mean = torch.randn(5, dtype=torch.complex128, generator=rng)
    root = torch.randn(5, 5, dtype=torch.complex128, generator=rng)
    covariance = root @ root.mH + torch.eye(5) * .3
    blocks = [PilotBlock(torch.randn(rows, 5, dtype=torch.complex128, generator=rng),
                         torch.randn(rows, dtype=torch.complex128, generator=rng), noise, index)
              for rows, noise, index in ((3, .2, 1), (7, .7, 3), (4, .1, 4))]
    return mean, covariance, blocks


def test_static_batch_recursive_and_arbitrary_row_coordinates_are_equivalent():
    mean, covariance, blocks = _problem()
    batch, cov = joint_posterior(mean, covariance, blocks)
    recursive, recursive_cov = recursive_posterior(mean, covariance, blocks)
    moved, moved_cov = joint_posterior(mean, covariance,
        [transport_block(block, torch.linspace(-.5, 1., block.target.numel())) for block in blocks])
    torch.testing.assert_close(batch, recursive, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov, recursive_cov, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(batch, moved, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(cov, moved_cov, atol=1e-12, rtol=1e-12)


def test_repeated_prior_matches_sum_of_unnormalized_single_frame_objectives():
    mean, covariance, blocks = _problem()
    # 不等行数、不等噪声时也必须按未归一化似然与一次先验求和。
    joint, _ = joint_posterior(mean, covariance, blocks)
    repeated, _ = joint_posterior(mean, covariance, blocks, prior_copies=len(blocks))
    precision = torch.linalg.inv(covariance)
    gram = len(blocks) * precision + sum(block.information()[0] for block in blocks)
    target = len(blocks) * precision @ mean + sum(block.information()[1] for block in blocks)
    torch.testing.assert_close(repeated, torch.linalg.solve(gram, target))
    assert float((joint - repeated).norm()) > .01


def test_ar_prediction_with_skipped_frames_matches_full_trajectory_map():
    mean, covariance, blocks = _problem()
    rho = .93
    process = torch.diag(torch.tensor([.1, .2, .3, .4, .5], dtype=torch.complex128))
    actual, actual_cov = recursive_posterior(mean, covariance, blocks, rho=rho, process_covariance=process)
    # 直接构造整段轨迹正规方程，消元后的最后一个边缘须与滤波一致。
    width, count = mean.numel(), len(blocks)
    precision = torch.zeros(width * count, width * count, dtype=torch.complex128)
    information = torch.zeros(width * count, dtype=torch.complex128)
    prior_precision = torch.linalg.inv(covariance)
    precision[:width, :width] = prior_precision
    information[:width] = prior_precision @ mean
    for index, block in enumerate(blocks):
        region = slice(index * width, (index + 1) * width)
        gram, target = block.information()
        precision[region, region] += gram
        information[region] += target
        if index:
            previous = slice((index - 1) * width, index * width)
            gap = block.frame_index - blocks[index-1].frame_index
            transition = rho ** gap
            noise = process * (1-rho ** (2 * gap)) / (1-rho ** 2)
            weight = torch.linalg.inv(noise)
            precision[region, region] += weight
            precision[previous, previous] += transition ** 2 * weight
            precision[region, previous] -= transition * weight
            precision[previous, region] -= transition * weight
    reference = torch.linalg.solve(precision, information)
    reference_cov = torch.linalg.inv(precision)
    torch.testing.assert_close(actual, reference[-width:], atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(actual_cov, reference_cov[-width:, -width:], atol=1e-12, rtol=1e-12)


def test_duplicate_frames_and_invalid_noise_are_rejected():
    mean, covariance, blocks = _problem()
    with pytest.raises(ValueError, match="严格递增"):
        recursive_posterior(mean, covariance, blocks + blocks[-1:])
    with pytest.raises(ValueError, match="正有限"):
        joint_posterior(mean, covariance, [replace(blocks[0], noise_variance=0.)])


def test_actual_adapt_coordinates_ignore_data_reward_and_hidden_truth():
    from tests.test_pilot_transfer import _channel_memory_fixture

    _, model, context = _channel_memory_fixture()
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    original, original_scale = extract_adapt_blocks(model, context)
    frame = context.frame
    bits, symbols, rx = frame.bits.clone(), frame.tx_symbols.clone(), frame.rx_symbols.clone()
    hidden = ~frame.adapt_mask
    bits[hidden], symbols[hidden], rx[hidden] = ~bits[hidden], complex(float("nan"), 0), complex(float("nan"), 0)
    poisoned = replace(frame, bits=bits, tx_symbols=symbols, rx_symbols=rx,
                       true_cir=torch.full_like(context.condition.complex_cir, complex(float("nan"), 0)))
    altered, scale = extract_adapt_blocks(model, replace(context, frame=poisoned))
    for name in original:
        torch.testing.assert_close(original[name].design, altered[name].design, atol=0, rtol=0)
        torch.testing.assert_close(original[name].target, altered[name].target, atol=0, rtol=0)
    torch.testing.assert_close(original_scale, scale, atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name], atol=0, rtol=0)


def test_synthetic_level_b_geometry_and_coordinate_checks():
    from scripts.diagnose_joint_pilot_state import synthetic_mechanisms

    result = synthetic_mechanisms()
    assert result["complex_joint_rank"] == 117
    assert result["uncorrected_phase_cir_mse"] > result["once_prior_cir_mse"]
    assert result["repeated_vs_once_solution_norm"] > .01
    assert result["online_control_executed"] is False


def test_shared_single_frame_matches_existing_map_value_and_gradient():
    from evaluation.joint_pilot_state import SharedNeuralPilotObjective
    from tests.test_pilot_transfer import _channel_memory_fixture
    from training.pilot_transfer import ChannelStateSignalObjective

    _, model, context = _channel_memory_fixture()
    model.set_trainable_groups({"channel_state_memory"})
    with torch.no_grad():
        model.channel_state.memory.weight.fill_(.04)
    shared = SharedNeuralPilotObjective(model, [context])()
    existing = ChannelStateSignalObjective(model, include_prior=True)(context)
    torch.testing.assert_close(shared, existing, atol=2e-5, rtol=1e-6)
    a, = torch.autograd.grad(shared, model.channel_state.memory.weight)
    b, = torch.autograd.grad(existing, model.channel_state.memory.weight)
    torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-6)


def test_shared_candidate_uses_only_pilot_and_keeps_original_neural_state():
    from scripts.diagnose_shared_neural_pilot import shared_candidate
    from tests.test_pilot_transfer import _channel_memory_fixture
    from training.pilot_transfer import differentiable_adapt

    _, model, context = _channel_memory_fixture()
    context = replace(context, frame=replace(context.frame,
        rx_symbols=context.frame.rx_symbols + .2j))
    settings = {"inner_learning_rate": .02, "inner_steps": 3, "max_delta_norm": .8}
    original = {name: value.clone() for name, value in model.state_dict().items()}
    control, before, after = shared_candidate(model, [context], settings)
    fast, _ = differentiable_adapt(model, [context], learning_rate=.02, steps=3, max_delta_norm=.8,
        groups={"channel_state_memory"}, objective="channel_state_map", receiver_input="full_rx")
    from training.pilot_transfer import context_logits
    with torch.no_grad():
        torch.testing.assert_close(after, context_logits(model, context, fast), atol=2e-5, rtol=1e-5)
    frame = context.frame
    tx, bits = frame.tx_symbols.clone(), frame.bits.clone()
    tx[frame.data_mask], bits[frame.data_mask] = complex(float("nan"), 0), ~bits[frame.data_mask]
    changed, altered_before, altered_after = shared_candidate(model,
        [replace(context, frame=replace(frame, tx_symbols=tx, bits=bits))], settings)
    assert control == changed
    assert 0 < control["delta_norm"] <= .800001
    torch.testing.assert_close(before, altered_before, atol=0, rtol=0)
    torch.testing.assert_close(after, altered_after, atol=0, rtol=0)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)


def test_shared_objective_isolates_all_history_hidden_and_reward_labels():
    from evaluation.joint_pilot_state import SharedNeuralPilotObjective
    from tests.test_pilot_transfer import _channel_memory_fixture

    _, model, context = _channel_memory_fixture()
    second = replace(context, frame=replace(context.frame, frame_index=context.frame.frame_index+1))
    contexts = [context, second]
    clean = SharedNeuralPilotObjective(model, contexts)()
    poisoned = []
    for current in contexts:
        frame = current.frame
        symbols, rx = frame.tx_symbols.clone(), frame.rx_symbols.clone()
        hidden = ~frame.adapt_mask
        symbols[hidden], rx[hidden] = complex(float("nan"), 0), complex(float("nan"), 0)
        poisoned.append(replace(current, frame=replace(frame, tx_symbols=symbols, rx_symbols=rx)))
    altered = SharedNeuralPilotObjective(model, poisoned)()
    torch.testing.assert_close(clean, altered, atol=0, rtol=0)
    with pytest.raises(ValueError, match="相邻"):
        SharedNeuralPilotObjective(model, contexts[::-1])
