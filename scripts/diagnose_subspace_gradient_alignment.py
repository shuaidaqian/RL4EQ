# -*- coding: utf-8 -*-
"""只读测量检测子空间Adapt、验证Pilot与Data梯度对齐。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.detection_subspace import DetectionSubspace


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = left.norm() * right.norm()
    if not bool(torch.isfinite(denominator)) or float(denominator) <= 1e-20:
        return None
    value = torch.dot(left, right) / denominator
    return float(value) if bool(torch.isfinite(value)) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def parse_temperatures(value: str) -> list[float]:
    try:
        temperatures = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("Adapt温度必须是逗号分隔的正有限数。") from exc
    if not temperatures or any(not torch.isfinite(torch.tensor(item)) or item <= 0 for item in temperatures):
        raise ValueError("Adapt温度必须是逗号分隔的正有限数。")
    if len(set(temperatures)) != len(temperatures):
        raise ValueError("Adapt温度不能重复。")
    return temperatures


def _loss_gradient(model: DetectionSubspace, observation, coefficients: torch.Tensor,
                   role: str, data_row: dict | None = None, *,
                   adapt_temperature: float = 1.0) -> tuple[float, torch.Tensor]:
    logits = model.logits(observation, coefficients)
    if role == "adapt":
        mask, labels = observation.adapt_mask, observation.adapt_labels
        if not torch.isfinite(torch.tensor(adapt_temperature)) or adapt_temperature <= 0:
            raise ValueError("Adapt温度必须为正有限数。")
        logits = logits / adapt_temperature
    elif role == "validation":
        mask, labels = observation.validation_mask, observation.validation_labels
    elif role == "data":
        if data_row is None:
            raise ValueError("Data梯度必须绑定离线评分封套。")
        mask = data_row["data_mask"].to(logits.device)
        labels = data_row["bits"].to(logits.device).float()
    else:
        raise ValueError("未知梯度目标。")
    if not bool(mask.any()):
        raise ValueError(f"{role}目标缺少有效样本。")
    loss = F.binary_cross_entropy_with_logits(logits[mask].float(), labels[mask].float())
    gradient, = torch.autograd.grad(loss, coefficients)
    if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(gradient).all()):
        raise FloatingPointError(f"{role}梯度非有限。")
    return float(loss.detach()), gradient.detach()


def analyze_sessions(sessions: list[dict], trajectory_rows: list[dict], checkpoint: Path,
                     *, device: str = "cpu", adapt_temperature: float = 1.0) -> dict:
    if not sessions:
        raise ValueError("缺少评估特征会话。")
    by_key = {(int(row["seed"]), int(row["snr_db"]), int(row["frame"])): row
              for row in trajectory_rows if row.get("training") == "B_post_adapt"}
    if len(by_key) != len(trajectory_rows) // 2:
        raise ValueError("评估轨迹缺少唯一B臂逐帧记录。")
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    dimension = int(payload["dimension"])
    model = DetectionSubspace(dimension, architecture=payload["architecture"],
        learned_gradient_transport=payload.get("learned_gradient_transport", False),
        learned_context_update=payload.get("learned_context_update", False)).to(device)
    model.use_context_update = payload.get("use_context_update", model.learned_context_update)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    rows = []
    for session in sessions:
        seed, snr = int(session["seed"]), int(session["snr_db"])
        origin = model.coefficients.detach().clone()
        current = origin
        history = []
        for index, frame in enumerate(session["frames"]):
            identity = frame["identity"]
            key = (seed, snr, int(identity["frame"]))
            if key not in by_key:
                raise ValueError("特征缓存与归档B臂轨迹帧不匹配。")
            logged = by_key[key]
            if identity.get("rx_sha256") != logged.get("rx_sha256") or identity.get("lookahead_rx_sha256") != logged.get("lookahead_rx_sha256"):
                raise ValueError("特征缓存与归档轨迹的接收波形指纹不一致。")
            if index:
                current = torch.tensor(logged_previous["coefficient_values"], device=device, dtype=origin.dtype)
            history.append(frame["observation"])
            history = history[-int(session.get("inner_history_frames", 8)):]
            current = current.detach().clone().requires_grad_(True)
            adapt_terms = [_loss_gradient(model, observation, current, "adapt",
                             adapt_temperature=adapt_temperature)[1] for observation in history]
            adapt_gradient = torch.stack(adapt_terms).mean(dim=0)
            validation_loss, validation_gradient = _loss_gradient(
                model, frame["observation"], current, "validation")
            data_loss, data_gradient = _loss_gradient(
                model, frame["observation"], current, "data", frame["offline"])
            rows.append({
                "seed": seed,
                "snr_db": snr,
                "frame": int(identity["frame"]),
                "adapt_history_frames": len(history),
                "adapt_gradient_norm": float(adapt_gradient.norm()),
                "validation_gradient_norm": float(validation_gradient.norm()),
                "data_gradient_norm": float(data_gradient.norm()),
                "adapt_validation_cosine": _cosine(adapt_gradient, validation_gradient),
                "adapt_data_cosine": _cosine(adapt_gradient, data_gradient),
                "validation_data_cosine": _cosine(validation_gradient, data_gradient),
                "validation_bce": validation_loss,
                "offline_data_bce": data_loss,
                "data_gradient_used_for_control": False,
            })
            logged_previous = logged

    session_rows: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for row in rows:
        session_rows[(row["snr_db"], row["seed"])].append(row)
    sessions_summary = []
    fields = ("adapt_gradient_norm", "validation_gradient_norm", "data_gradient_norm",
              "adapt_validation_cosine", "adapt_data_cosine", "validation_data_cosine")
    for (snr, seed), values in sorted(session_rows.items()):
        summary = {"snr_db": snr, "seed": seed, "frames": len(values)}
        for field in fields:
            summary[field + "_mean"] = _mean([row[field] for row in values if row[field] is not None])
        sessions_summary.append(summary)
    by_snr = []
    for snr in sorted({row["snr_db"] for row in sessions_summary}):
        items = [row for row in sessions_summary if row["snr_db"] == snr]
        by_snr.append({"snr_db": snr, "independent_seed_sessions": len(items),
                       **{field + "_mean_of_seed_means": _mean([
                           row[field + "_mean"] for row in items if row[field + "_mean"] is not None
                       ]) for field in fields}})
    return {
        "schema_version": "subspace-gradient-alignment-v1",
        "diagnostic_only": True,
        "data_gradient_used_for_control": False,
        "gradient_parameters": "第六轮实际在线更新的8维系数；基网络与来源网络冻结",
        "gradient_definition": "当前更新前系数状态；Adapt为最近最多8帧Pilot BCE均值；验证为当前帧Reward Pilot BCE；Data为当前帧离线标签BCE",
        "adapt_logit_temperature": adapt_temperature,
        "independent_unit": "seed-session; 每SNR仅3个重复开发seed",
        "by_snr": by_snr,
        "sessions": sessions_summary,
        "frames": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--adapt-temperatures", default="1.0",
                        help="只缩放Adapt BCE logits的全局温度，逗号分隔；验证与Data目标不变")
    args = parser.parse_args()
    paths = {name: Path(getattr(args, name.replace("-", "_"))) for name in
             ("feature-cache", "trajectory", "checkpoint")}
    output = Path(args.output)
    if output.exists():
        raise ValueError("诊断输出已存在，请提供新路径。")
    sessions = torch.load(paths["feature-cache"], map_location=args.device, weights_only=False)
    with gzip.open(paths["trajectory"], "rt", encoding="utf-8") as handle:
        trajectory_rows = [json.loads(line) for line in handle if line.strip()]
    temperatures = parse_temperatures(args.adapt_temperatures)
    analyses = [analyze_sessions(sessions, trajectory_rows, paths["checkpoint"],
                  device=args.device, adapt_temperature=temperature)
                for temperature in temperatures]
    baseline_index = temperatures.index(1.0) if 1.0 in temperatures else 0
    result = analyses[baseline_index]
    result["temperature_sweep"] = [
        {"adapt_logit_temperature": temperature, "by_snr": analysis["by_snr"],
         "sessions": analysis["sessions"]}
        for temperature, analysis in zip(temperatures, analyses)
    ]
    result["temperature_sweep_order"] = temperatures
    result["inputs"] = {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["by_snr"], ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
