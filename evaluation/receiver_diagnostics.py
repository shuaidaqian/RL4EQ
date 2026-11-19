# -*- coding: utf-8 -*-
"""严格在接收完成后评分的状态尺度与 soft 可靠性诊断。"""

import torch
import torch.nn.functional as F


def receiver_diagnostics(logits, labels, data_mask, estimate, truth, *, confidence_threshold=.95):
    """输入含离线真值，禁止从接收机或在线更新路径调用。"""
    selected = logits[data_mask].float()
    target = labels[data_mask].bool()
    probability = selected.sigmoid()
    confident = selected.abs().sigmoid() >= confidence_threshold
    wrong = (selected>=0) != target
    energy = estimate.abs().square().sum().clamp_min(1e-12)
    truth_energy = truth.abs().square().sum().clamp_min(1e-12)
    inner = torch.vdot(estimate.flatten(), truth.flatten())
    scale = inner/energy
    phase = inner/inner.abs().clamp_min(1e-12)
    return dict(
        aligned_nmse_diagnostic=float((truth-scale*estimate).abs().square().sum()/truth_energy),
        phase_only_nmse_diagnostic=float((truth-phase*estimate).abs().square().sum()/truth_energy),
        decode_cir_norm=float(estimate.norm()), true_cir_norm_diagnostic=float(truth.norm()),
        data_bce=float(F.binary_cross_entropy_with_logits(selected, target.float())),
        data_brier=float((probability-target.float()).square().mean()),
        confident_data_count=int(confident.sum()), confident_wrong_data_count=int((confident & wrong).sum()),
        confidence_threshold=confidence_threshold,
    )
