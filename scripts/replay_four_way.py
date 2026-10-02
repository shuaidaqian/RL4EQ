"""运行 Frozen、Pilot 状态、PEFT 和联合在线适配的统一配对 replay。"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.replay_abc import _read_jsonl, _write_config, _write_jsonl, replay_pair_key, relabel_rows


@dataclass(frozen=True)
class FourWaySpec:
    """一个四路消融方法的运行时定义。"""

    name: str
    source_method: str
    config: dict[str, Any]
    cir_update: str
    scheduler: str = "fixed"


def build_four_way_specs(base_config: dict[str, Any]) -> tuple[FourWaySpec, ...]:
    """构造四路定义，统一使用 SGD PEFT，禁止在线读取数据标签。"""

    def config_for(*, groups: list[str], cir_update: str) -> dict[str, Any]:
        config = copy.deepcopy(base_config)
        config["online_adaptation_algorithm"] = "sgd"
        config["online_adaptation_groups"] = list(groups)
        config["online_adaptation_candidates"] = [
            {
                "name": "four_way_candidate",
                "groups": list(groups),
                "learning_rate_scale": 1.0,
                "steps": int(config.get("online_adaptation_steps", 1)),
                "max_delta_scale": 1.0,
            }
        ] if groups else []
        config["data_labels_used_online"] = False
        config["four_way_cir_update"] = cir_update
        return config

    return (
        FourWaySpec("Frozen Offline NN", "Frozen Offline NN", config_for(groups=[], cir_update="fixed"), "fixed"),
        FourWaySpec("Pilot State Only", "Pilot CIR only", config_for(groups=[], cir_update="pilot_sparse"), "pilot_sparse"),
        FourWaySpec("PEFT Only", "Pilot-Driven Online Adaptation", config_for(groups=["phase"], cir_update="fixed"), "fixed"),
        FourWaySpec("Joint State + PEFT", "Pilot-Driven Online Adaptation", config_for(groups=["phase"], cir_update="pilot_sparse"), "pilot_sparse"),
    )


def _run_spec(
    spec: FourWaySpec,
    config_path: Path,
    pretrained: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "compare.py",
        "--config", str(config_path),
        "--pretrained", str(pretrained),
        "--methods", spec.source_method,
        "--delays", *[str(value) for value in args.delays],
        "--snrs", *[str(value) for value in args.snrs],
        "--num-seeds", str(args.num_seeds),
        "--frames", str(args.frames),
        "--pilot-total", str(args.pilot_total),
        "--reward-pilot-total", str(args.reward_pilot_total),
        "--pilot-layout", args.pilot_layout,
        "--update-interval", str(args.update_interval),
        "--scheduler", "fixed",
        "--cir-update", spec.cir_update,
        "--online-condition-source", "pilot_cir_phase",
        "--output-dir", str(output_dir),
        "--device", args.device,
    ]
    print("运行：" + " ".join(command))
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"compare.py 运行失败：{spec.name}，退出码={completed.returncode}")
    rows = relabel_rows(_read_jsonl(output_dir / "frame_metrics.jsonl"), spec.name, spec.source_method)
    for row in rows:
        row["four_way_definition"] = {
            "cir_update": spec.cir_update,
            "peft_groups": list(spec.config.get("online_adaptation_groups", [])),
            "scheduler": spec.scheduler,
        }
    return rows


def run_four_way(args: argparse.Namespace) -> dict[str, Any]:
    """执行四路配对实验并写出逐帧及汇总结果。"""

    base_config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pretrained = Path(args.pretrained)
    if not pretrained.exists():
        raise FileNotFoundError(f"checkpoint 不存在：{pretrained}")
    target = Path(args.output_dir)
    target.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    specs = build_four_way_specs(base_config)
    for index, spec in enumerate(specs):
        config_path = target / f"config_{index}.json"
        _write_config(config_path, spec.config)
        all_rows.extend(_run_spec(spec, config_path, pretrained, target / f"method_{index}", args))

    by_method: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    for row in all_rows:
        by_method.setdefault(str(row["method"]), {})[replay_pair_key(row)] = row
    reference_keys = set(by_method["Frozen Offline NN"])
    if any(set(rows) != reference_keys for name, rows in by_method.items() if name != "Frozen Offline NN"):
        raise ValueError("四路方法与 Frozen 的配对键不一致。")

    summary: dict[str, Any] = {"schema_version": "four-way-pilot-replay-v1", "methods": {}}
    for name, rows in by_method.items():
        summary["methods"][name] = {
            "count": len(rows),
            "mean_ber_data": mean(float(row["ber_data"]) for row in rows.values()),
            "pretrained_loaded_all": all(bool(row.get("pretrained_loaded", False)) for row in rows.values()),
            "cir_update_applied": sum(bool(row.get("cir_update_applied", False)) for row in rows.values()),
            "peft_update_applied": sum(bool(row.get("peft_update_applied", False)) for row in rows.values()),
            "adaptation_accepted": sum(bool(row.get("adaptation_accepted", False)) for row in rows.values()),
            "rollback": sum(bool(row.get("rollback", False)) for row in rows.values()),
            "data_labels_used_online": any(bool(row.get("data_labels_used_online", True)) for row in rows.values()),
        }
    summary["protocol"] = {
        "config": str(Path(args.config)),
        "pretrained": str(pretrained),
        "frames": int(args.frames),
        "num_seeds": int(args.num_seeds),
        "delays": [int(value) for value in args.delays],
        "snrs": [float(value) for value in args.snrs],
        "pilot_total": int(args.pilot_total),
        "reward_pilot_total": int(args.reward_pilot_total),
        "pilot_layout": str(args.pilot_layout),
        "update_interval": int(args.update_interval),
        "data_labels_used_online": False,
    }
    _write_jsonl(target / "frame_metrics.jsonl", all_rows)
    (target / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="四路 Pilot-only 在线消融 replay")
    parser.add_argument("--config", default="configs/eme_long_memory_v2.json")
    parser.add_argument("--pretrained", default="pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt")
    parser.add_argument("--output-dir", default="logs/four_way_replay_smoke")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--delays", nargs="+", type=int, default=[116])
    parser.add_argument("--snrs", nargs="+", type=float, default=[0, 5, 10, 15])
    parser.add_argument("--pilot-total", type=int, default=256)
    parser.add_argument("--reward-pilot-total", type=int, default=32)
    parser.add_argument("--pilot-layout", default="prefix")
    parser.add_argument("--update-interval", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


if __name__ == "__main__":
    run_four_way(_parse_args())
