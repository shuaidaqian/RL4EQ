"""运行 A/B/C 统一小样本 replay 并生成配对结果。"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


@dataclass(frozen=True)
class ReplaySpec:
    """一个 replay 候选的方法映射和运行时配置。"""

    name: str
    source_method: str
    config: dict[str, Any]
    cir_update: str
    scheduler: str = "fixed"


REPLAY_METHODS = (
    "ABC-A Pilot State Gate",
    "ABC-B Pilot RLS Residual",
    "ABC-C Physics Unfolded",
)
REPLAY_METHOD_CODES = ("A", "B", "C")
REPLAY_METHOD_BY_CODE = dict(zip(REPLAY_METHOD_CODES, REPLAY_METHODS))


def _normalize_candidates(candidates: tuple[str, ...] | list[str] | None) -> tuple[str, ...]:
    """规范化候选代码，并保持调用方给出的顺序。"""

    selected = REPLAY_METHOD_CODES if candidates is None else tuple(
        str(value).strip().upper() for value in candidates
    )
    if not selected:
        raise ValueError("至少选择一个 replay 候选（A、B 或 C）。")
    unknown = [value for value in selected if value not in REPLAY_METHOD_BY_CODE]
    if unknown:
        raise ValueError(f"未知 replay 候选：{unknown}；可选值为 A、B、C。")
    duplicates = sorted({value for value in selected if selected.count(value) > 1})
    if duplicates:
        raise ValueError(f"replay 候选不能重复：{duplicates}")
    return selected


def build_replay_specs(
    base_config: dict[str, Any],
    candidates: tuple[str, ...] | list[str] | None = None,
) -> tuple[ReplaySpec, ...]:
    """把选定的 A/B/C 定义映射到现有 compare.py 方法。"""

    selected = _normalize_candidates(candidates)
    specs: list[ReplaySpec] = []
    for code in selected:
        name = REPLAY_METHOD_BY_CODE[code]
        config = copy.deepcopy(base_config)
        if name == "ABC-B Pilot RLS Residual":
            config["online_adaptation_algorithm"] = "rls"
            config["online_rls_condition_source"] = "pilot_cir_phase"
        if name == "ABC-C Physics Unfolded":
            model_config = dict(config.get("model", {}))
            # 关闭神经残差，只保留 checkpoint 中的物理 warm-start 展开和
            # 已冻结的相位先验；在线阶段不更新任何 PEFT 参数。
            model_config["neural_residual_scale"] = 0.0
            config["model"] = model_config
        if code == "B":
            specs.append(
                ReplaySpec(
                    name=name,
                    source_method="Pilot-Driven Online Adaptation",
                    config=config,
                    cir_update="fixed",
                )
            )
        else:
            # C 复用 Pilot CIR only 的状态跟踪分支，再关闭神经 residual，
            # 才能同时满足“Pilot 驱动 CIR 更新”和“无在线 PEFT 更新”。
            specs.append(
                ReplaySpec(
                    name=name,
                    source_method="Pilot CIR only",
                    config=config,
                    cir_update="pilot_sparse",
                )
            )
    return tuple(specs)


def replay_pair_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """返回跨方法对齐所需的 seed/frame/协议键。"""

    return (
        str(row.get("level", "")),
        str(row.get("profile_name", "")),
        str(row.get("impairment_profile", "")),
        int(row["delay"]),
        float(row["snr_db"]),
        int(row["seed"]),
        int(row["frame"]),
        int(row["pilot_total"]),
        int(row.get("reward_pilot_total", 0)),
        str(row["pilot_layout"]),
    )


def relabel_rows(
    rows: list[dict[str, Any]],
    replay_name: str,
    source_method: str,
) -> list[dict[str, Any]]:
    """保留原始方法信息，同时把结果改成 replay 候选名称。"""

    relabeled: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        if str(row.get("method")) != str(source_method):
            raise ValueError(
                f"replay 输出方法不匹配：期望 {source_method}，实际 {row.get('method')}。"
            )
        row["source_method"] = str(source_method)
        row["method"] = str(replay_name)
        row["replay_candidate"] = str(replay_name)
        # compare.py 已经保证这一字段为 False；这里再次检查，防止后续方法越界。
        row["data_labels_used_online"] = bool(row.get("data_labels_used_online", False))
        if row["data_labels_used_online"]:
            raise ValueError(f"{replay_name} 违反 Data 标签隔离。")
        relabeled.append(row)
    return relabeled


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"无法解析 {path}:{line_number}") from exc
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _paired_summary(
    rows: list[dict[str, Any]],
    frozen_name: str = "Frozen Offline NN",
) -> dict[str, Any]:
    """按 SNR 汇总 Frozen 与候选方法的逐帧配对差。"""

    frozen = {
        replay_pair_key(row): row
        for row in rows
        if row.get("method") == frozen_name
    }
    candidates = sorted(
        {
            str(row["method"])
            for row in rows
            if row.get("method") != frozen_name
        }
    )
    per_candidate: dict[str, Any] = {}
    for candidate in candidates:
        candidate_rows = {
            replay_pair_key(row): row
            for row in rows
            if row.get("method") == candidate
        }
        paired: list[dict[str, Any]] = []
        for key in sorted(set(frozen) & set(candidate_rows), key=str):
            frozen_row = frozen[key]
            candidate_row = candidate_rows[key]
            paired.append(
                {
                    "key": list(key),
                    "snr_db": float(candidate_row["snr_db"]),
                    "seed": int(candidate_row["seed"]),
                    "frame": int(candidate_row["frame"]),
                    "ber_frozen": float(frozen_row["ber_data"]),
                    "ber_candidate": float(candidate_row["ber_data"]),
                    "ber_frozen_minus_candidate": float(
                        frozen_row["ber_data"] - candidate_row["ber_data"]
                    ),
                }
            )
        by_snr: dict[str, Any] = {}
        for snr in sorted({float(item["snr_db"]) for item in paired}):
            values = [
                item["ber_frozen_minus_candidate"]
                for item in paired
                if float(item["snr_db"]) == snr
            ]
            candidate_ber = [
                item["ber_candidate"]
                for item in paired
                if float(item["snr_db"]) == snr
            ]
            frozen_ber = [
                item["ber_frozen"]
                for item in paired
                if float(item["snr_db"]) == snr
            ]
            by_snr[str(snr)] = {
                "n_frames": len(values),
                "frozen_ber_mean": mean(frozen_ber),
                "candidate_ber_mean": mean(candidate_ber),
                "paired_gain_mean": mean(values),
                "paired_gain_median": median(values),
                "candidate_better_fraction": mean(value > 0.0 for value in values),
            }
        per_candidate[candidate] = {
            "n_paired_frames": len(paired),
            "per_snr": by_snr,
            "paired_frames": paired,
        }
    return {
        "schema_version": "abc-pilot-replay-summary-v1",
        "frozen_reference": frozen_name,
        "candidate_methods": candidates,
        "per_candidate": per_candidate,
    }


def _write_config(path: Path, config: dict[str, Any]) -> None:
    path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_compare(
    *,
    spec: ReplaySpec,
    config_path: Path,
    pretrained: Path,
    output_dir: Path,
    frames: int,
    num_seeds: int,
    delays: list[int],
    snrs: list[float],
    pilot_total: int,
    reward_pilot_total: int,
    pilot_layout: str,
    update_interval: int,
    device: str,
) -> list[dict[str, Any]]:
    """调用统一 compare.py，确保每个候选使用相同协议参数。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "compare.py",
        "--config",
        str(config_path),
        "--pretrained",
        str(pretrained),
        "--methods",
        spec.source_method,
        "--delays",
        *[str(value) for value in delays],
        "--snrs",
        *[str(value) for value in snrs],
        "--num-seeds",
        str(num_seeds),
        "--frames",
        str(frames),
        "--pilot-total",
        str(pilot_total),
        "--reward-pilot-total",
        str(reward_pilot_total),
        "--pilot-layout",
        pilot_layout,
        "--update-interval",
        str(update_interval),
        "--scheduler",
        spec.scheduler,
        "--cir-update",
        spec.cir_update,
        "--output-dir",
        str(output_dir),
        "--device",
        device,
    ]
    print("运行：" + " ".join(command))
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"compare.py 运行失败，候选={spec.name}，退出码={completed.returncode}")
    metrics_path = output_dir / "frame_metrics.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(f"缺少 compare 输出：{metrics_path}")
    return relabel_rows(_read_jsonl(metrics_path), spec.name, spec.source_method)


def run_replay(args: argparse.Namespace) -> dict[str, Any]:
    """执行 Frozen 参考和 A/B/C 三个候选。"""

    base_config_path = Path(args.config)
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    pretrained = Path(args.pretrained)
    if not pretrained.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{pretrained}")
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)

    frozen_spec = ReplaySpec(
        name="Frozen Offline NN",
        source_method="Frozen Offline NN",
        config=copy.deepcopy(base_config),
        cir_update="fixed",
    )
    frozen_config_path = target / "config_frozen.json"
    _write_config(frozen_config_path, frozen_spec.config)
    all_rows = _run_compare(
        spec=frozen_spec,
        config_path=frozen_config_path,
        pretrained=pretrained,
        output_dir=target / "frozen",
        frames=args.frames,
        num_seeds=args.num_seeds,
        delays=args.delays,
        snrs=args.snrs,
        pilot_total=args.pilot_total,
        reward_pilot_total=args.reward_pilot_total,
        pilot_layout=args.pilot_layout,
        update_interval=args.update_interval,
        device=args.device,
    )

    specs = build_replay_specs(base_config, candidates=args.candidates)
    for index, spec in enumerate(specs, start=1):
        config_path = target / f"config_{index}.json"
        _write_config(config_path, spec.config)
        all_rows.extend(
            _run_compare(
                spec=spec,
                config_path=config_path,
                pretrained=pretrained,
                output_dir=target / f"candidate_{index}",
                frames=args.frames,
                num_seeds=args.num_seeds,
                delays=args.delays,
                snrs=args.snrs,
                pilot_total=args.pilot_total,
                reward_pilot_total=args.reward_pilot_total,
                pilot_layout=args.pilot_layout,
                update_interval=args.update_interval,
                device=args.device,
            )
        )

    pair_keys_by_method: dict[str, set[tuple[Any, ...]]] = {}
    for row in all_rows:
        pair_keys_by_method.setdefault(str(row["method"]), set()).add(replay_pair_key(row))
    frozen_keys = pair_keys_by_method["Frozen Offline NN"]
    for method, keys in pair_keys_by_method.items():
        if method == "Frozen Offline NN":
            continue
        if keys != frozen_keys:
            raise ValueError(
                f"{method} 与 Frozen 的配对键不一致："
                f"candidate_only={len(keys - frozen_keys)}, frozen_only={len(frozen_keys - keys)}"
            )

    _write_jsonl(target / "frame_metrics.jsonl", all_rows)
    summary = _paired_summary(all_rows)
    summary.update(
        {
            "protocol": {
                "config": str(base_config_path),
                "pretrained": str(pretrained),
                "frames": int(args.frames),
                "num_seeds": int(args.num_seeds),
                "delays": [int(value) for value in args.delays],
                "snrs": [float(value) for value in args.snrs],
                "candidates": list(_normalize_candidates(args.candidates)),
                "pilot_total": int(args.pilot_total),
                "reward_pilot_total": int(args.reward_pilot_total),
                "pilot_layout": str(args.pilot_layout),
                "update_interval": int(args.update_interval),
                "scheduler": "fixed",
                "data_labels_used_online": False,
            },
            "method_definitions": {
                spec.name: {
                    "source_method": spec.source_method,
                    "cir_update": spec.cir_update,
                    "scheduler": spec.scheduler,
                    "config_path": str(target / f"config_{index}.json"),
                }
                for index, spec in enumerate(specs, start=1)
            },
        }
    )
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"已保存 replay（候选={','.join(_normalize_candidates(args.candidates))}）：{target}")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A/B/C Pilot-only 统一小样本 replay")
    parser.add_argument("--config", default="configs/eme_long_memory_v2.json")
    parser.add_argument(
        "--pretrained",
        default="pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt",
    )
    parser.add_argument("--output-dir", default="logs/abc_replay_smoke")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--delays", nargs="+", type=int, default=[116])
    parser.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--reward-pilot-total", type=int, default=32)
    parser.add_argument("--pilot-layout", default="prefix")
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument(
        "--candidates",
        nargs="+",
        choices=list(REPLAY_METHOD_CODES),
        default=list(REPLAY_METHOD_CODES),
        help="本轮运行的 replay 候选代码；Frozen 参考始终保留。",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run_replay(_parse_args())
