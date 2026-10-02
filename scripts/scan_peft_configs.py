"""扫描无 RL 的在线 PEFT 参数组和强度。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
BASE = ROOT / "configs" / "eme_long_memory_v2.json"
CHECKPOINT = ROOT / "pretrained" / "eme_bce_all_32_20260905_pilot256" / "model_best.pt"


def main() -> None:
    base = json.loads(BASE.read_text(encoding="utf-8"))
    scans = (
        ("phase_film_steps4", ["phase", "conditioner_film"], 0.001, 4, 0.02),
        ("phase_head_steps4", ["phase", "head"], 0.001, 4, 0.02),
        ("phase_film_head_steps4", ["phase", "conditioner_film", "head"], 0.001, 4, 0.02),
    )
    target = ROOT / "logs" / "peft_scan_20261002"
    target.mkdir(parents=True, exist_ok=True)
    for name, groups, lr, steps, threshold in scans:
        config = dict(base)
        config["online_adaptation_algorithm"] = "sgd"
        config["online_adaptation_groups"] = groups
        config["online_adaptation_candidates"] = [{
            "name": name,
            "groups": groups,
            "learning_rate_scale": 1.0,
            "steps": steps,
            "max_delta_scale": 1.0,
        }]
        config["online_adaptation_freeze_below_snr_db"] = None
        config["online_adaptation_min_reward_improvement"] = 0.0
        config["online_adaptation_relative_min_reward_improvement"] = 0.01
        config["online_adaptation_max_delta_norm"] = 0.02
        config["online_hard_example_weighting"] = True
        config["online_hard_example_temperature"] = 0.5
        path = target / f"{name}.json"
        path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        out = target / name
        command = [
            PYTHON, "compare.py", "--config", str(path), "--pretrained", str(CHECKPOINT),
            "--methods", "Frozen Offline NN", "Pilot-Driven Online Adaptation",
            "--state-split", "heldout_edge", "--impairment-profile", "cfo_phase_tiny",
            "--delays", "116", "--snrs", "10", "15", "--num-seeds", "1", "--frames", "8",
            "--pilot-total", "256", "--reward-pilot-total", "32", "--pilot-layout", "prefix",
            "--scheduler", "fixed", "--update-interval", "1", "--cir-update", "fixed",
            "--online-condition-source", "pilot_cir_phase", "--online-learning-rate", str(lr),
            "--online-steps", str(steps), "--online-min-reward-improvement", str(threshold),
            "--online-min-reward-improvement", "0.0",
            "--online-max-delta-norm", "0.02",
            "--online-relative-min-reward-improvement", "0.01",
            "--output-dir", str(out), "--device", "cuda",
        ]
        print("运行：" + " ".join(command))
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
