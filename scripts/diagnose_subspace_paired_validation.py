# -*- coding: utf-8 -*-
"""只读重建动作候选，评估逐符号配对验证Pilot损失及其区块不确定性。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.subspace_development import ROOT, make_subspace, read_manifest, sha256
from training.deterministic_adaptation import DeterministicSettings
from training.subspace_adaptation import update_subspace


def paired_improvement(before_logits, after_logits, labels, mask):
    """返回每个验证符号的 BCE 改善量，正数代表候选更好。"""
    before = F.binary_cross_entropy_with_logits(before_logits[mask].float(), labels[mask].float(), reduction="none")
    after = F.binary_cross_entropy_with_logits(after_logits[mask].float(), labels[mask].float(), reduction="none")
    return before - after


def moving_block_interval(values: torch.Tensor, *, block_length: int = 4,
                          resamples: int = 2000, seed: int = 0):
    """以循环移动区块bootstrap保留相邻符号相关，返回均值和95%百分位区间。"""
    values = values.detach().float().cpu().flatten()
    n = values.numel()
    if n == 0 or block_length < 1 or block_length > n or resamples < 100:
        raise ValueError("验证序列长度、区块长度或重采样次数非法。")
    blocks_needed = (n + block_length - 1) // block_length
    generator = torch.Generator().manual_seed(seed)
    starts = torch.randint(n, (resamples, blocks_needed), generator=generator)
    offsets = torch.arange(block_length)
    indices = (starts.unsqueeze(-1) + offsets) % n
    samples = values[indices.reshape(resamples, -1)[:, :n]].mean(dim=1)
    interval = torch.quantile(samples, torch.tensor([0.025, 0.975]))
    return {"mean": float(values.mean()), "lower95": float(interval[0]),
            "upper95": float(interval[1]), "block_length": block_length,
            "resamples": resamples}


def _bce_gain(logits_before, logits_after, labels, mask):
    before = F.binary_cross_entropy_with_logits(logits_before[mask].float(), labels[mask].float())
    after = F.binary_cross_entropy_with_logits(logits_after[mask].float(), labels[mask].float())
    return float((before - after).detach())


def analyze(manifest_path: Path, training_dir: Path, trajectory_path: Path,
            *, device: str = "cpu", block_lengths=(2, 4, 8)):
    spec = read_manifest(manifest_path)
    root = training_dir
    sessions = torch.load(root / "evaluation_features.pt", map_location=device, weights_only=False)
    payload = torch.load(root / "B_post_adapt.pt", map_location=device, weights_only=False)
    model = make_subspace(int(payload["dimension"]), int(spec["initialization_seed"]),
                          device, payload["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    with gzip.open(trajectory_path, "rt", encoding="utf-8") as handle:
        trajectory = [json.loads(line) for line in handle if line.strip()]
    by_key = {(int(row["seed"]), int(row["snr_db"]), int(row["frame"])): row
              for row in trajectory if row.get("training") == "B_post_adapt"}
    expected = sum(len(session["frames"]) for session in sessions)
    if len(by_key) != expected or len(by_key) != sum(row.get("training") == "B_post_adapt" for row in trajectory):
        raise ValueError("B臂轨迹帧数或唯一键与缓存会话不匹配。")
    settings = DeterministicSettings(**spec["update"])
    output_rows = []
    for session in sessions:
        seed, snr = int(session["seed"]), int(session["snr_db"])
        origin = model.coefficients.detach().clone()
        current = origin.clone()
        history = []
        previous_logged = None
        for index, frame in enumerate(session["frames"]):
            observation = frame["observation"]
            identity = frame["identity"]
            key = (seed, snr, int(identity["frame"]))
            if key not in by_key:
                raise ValueError("评估特征缓存和轨迹帧身份不匹配。")
            logged = by_key[key]
            if (identity.get("rx_sha256") != logged.get("rx_sha256") or
                    identity.get("lookahead_rx_sha256") != logged.get("lookahead_rx_sha256")):
                raise ValueError("波形指纹不匹配，拒绝重建更新候选。")
            if previous_logged is not None:
                current = torch.tensor(previous_logged["coefficient_values"], device=device,
                                       dtype=origin.dtype)
            history.append(observation)
            history = history[-int(session.get("inner_history_frames", 8)):]
            before = current.detach().clone()
            candidate, retained, control = update_subspace(
                model, observation, before, origin, settings, adapt_history=history)
            if abs(float(control["validation_loss_candidate"]) - float(logged["validation_loss_candidate"])) > 2e-6:
                raise ValueError("重建候选与归档验证损失不一致，拒绝配对损失分析。")
            basis = model.basis(observation)
            before_logits = observation.base_logits + basis @ before
            candidate_logits = observation.base_logits + basis @ candidate
            retained_logits = observation.base_logits + basis @ retained
            paired = paired_improvement(before_logits, candidate_logits,
                observation.validation_labels, observation.validation_mask)
            intervals = {str(length): moving_block_interval(paired,
                block_length=length, seed=seed * 100000 + (snr + 1) * 1000 + index * 10 + length)
                for length in block_lengths}
            offline = frame["offline"]
            data_mask = offline["data_mask"].to(device)
            data_labels = offline["bits"].to(device).float()
            data_before = observation.base_logits + basis @ before
            data_candidate = observation.base_logits + basis @ candidate
            data_retained = observation.base_logits + basis @ retained
            pilot_candidate_gain = _bce_gain(before_logits, candidate_logits,
                observation.validation_labels, observation.validation_mask)
            validation_hard_ok = control["validation_ber_candidate"] <= control["validation_ber_before"]
            row = {
                "seed": seed, "snr_db": snr, "frame": int(identity["frame"]),
                "original_mean_gate_accept": bool(control["candidate_accepted"]),
                "validation_hard_ber_ok": bool(validation_hard_ok),
                "validation_symbols": int(observation.validation_mask.sum()),
                "paired_pilot_bce_gain_mean": pilot_candidate_gain,
                "paired_block_bootstrap": intervals,
                "candidate_data_bce_gain_posthoc": _bce_gain(data_before, data_candidate, data_labels, data_mask),
                "retained_data_bce_gain_posthoc": _bce_gain(data_before, data_retained, data_labels, data_mask),
                "candidate_data_ber_gain_posthoc": float(
                    (((data_before[data_mask] >= 0) != data_labels[data_mask]).float().mean()) -
                    (((data_candidate[data_mask] >= 0) != data_labels[data_mask]).float().mean())),
                "data_used_for_gate": False,
            }
            for length, interval in intervals.items():
                row[f"gate_lb95_block{length}"] = bool(
                    interval["lower95"] > settings.min_validation_improvement and validation_hard_ok)
                row[f"gate_upper_nonharm_block{length}"] = bool(
                    interval["upper95"] >= 0 and pilot_candidate_gain > 0 and validation_hard_ok)
            output_rows.append(row)
            previous_logged = logged

    summaries = []
    for snr in sorted({row["snr_db"] for row in output_rows}):
        subset = [row for row in output_rows if row["snr_db"] == snr]
        summary = {"snr_db": snr, "frames": len(subset),
                   "original_accepts": sum(row["original_mean_gate_accept"] for row in subset)}
        for gate in ("original_mean_gate_accept", "gate_lb95_block2", "gate_lb95_block4",
                     "gate_lb95_block8", "gate_upper_nonharm_block2",
                     "gate_upper_nonharm_block4", "gate_upper_nonharm_block8"):
            chosen = [row for row in subset if row[gate]]
            summary[gate] = {
                "accepted": len(chosen),
                "mean_candidate_data_bce_gain_posthoc": (sum(r["candidate_data_bce_gain_posthoc"] for r in chosen) / len(chosen)) if chosen else None,
                "mean_retained_data_bce_gain_posthoc": (sum(r["retained_data_bce_gain_posthoc"] for r in chosen) / len(chosen)) if chosen else None,
                "harmful_candidate_fraction_posthoc": (sum(r["candidate_data_bce_gain_posthoc"] < 0 for r in chosen) / len(chosen)) if chosen else None,
            }
        summaries.append(summary)
    return {"schema_version": "subspace-paired-validation-diagnostic-v1",
        "role": "development_read_only_diagnostic", "heldout_confirmation": False,
        "data_used_for_gate": False,
        "method": "重建实际B臂每帧Adapt候选；逐验证Pilot符号配对BCE差；循环移动区块bootstrap估计不确定性；Data仅动作后用于规则诊断",
        "bootstrap_limit": "验证Pilot仅32符号，区块长度敏感性2/4/8仅为开发诊断；不是校准过的正式置信界",
        "inputs": {"manifest_sha256": sha256(manifest_path),
            "training_summary_sha256": sha256(root / "summary.json"),
            "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest()},
        "by_snr": summaries, "frames": output_rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("诊断输出已存在，拒绝覆盖。")
    result = analyze(Path(args.manifest), Path(args.training_dir), Path(args.trajectory), device=args.device)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["by_snr"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
