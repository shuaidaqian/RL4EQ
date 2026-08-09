# -*- coding: utf-8 -*-
"""RL4EQ 研究假设诊断入口。

该脚本只用于开发诊断，不产生正式论文矩阵。Data 标签只用于离线诊断报告，
不会进入在线 observation、reward、动作选择或 PPO 更新。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.cir_estimator import condition_from_cir
from baseline.block_equalizers import bit_error_rate
from env.comm_env import CommEnvConfig, CommunicationEnvironment
from evaluation.research_diagnostics import (
    apply_adapt_only_peft_update,
    evaluate_peft_window_candidates,
    evaluate_modulation_candidates,
    run_level_b_difficulty_scan,
    structured_modulation_candidates,
    summarize_reward_data_correlation,
    summarize_reward_selected_actions,
    summarize_reward_surrogates,
    summarize_windowed_reward_data_correlation,
    summarize_windowed_selected_actions,
)
from training.meta_training import _estimate_cir_from_known_frame
from training.rl_modulated_online import _build_equalizer, _load_model_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/continual_ppo.json")
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output-dir", default="logs/research_diagnostics")
    parser.add_argument("--delays", nargs="*", type=int, default=[20])
    parser.add_argument("--snrs", nargs="*", type=float, default=[10.0])
    parser.add_argument("--pilot-totals", nargs="*", type=int, default=[128])
    parser.add_argument("--pilot-layouts", nargs="*", default=["prefix"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[0])
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--peft-groups", nargs="*", default=["head", "adapter_lora"])
    parser.add_argument("--peft-steps", type=int, default=1)
    parser.add_argument("--peft-lr", type=float, default=1e-4)
    parser.add_argument("--action-space", choices=["focused_peft", "low_dim_modulation"], default="focused_peft")
    parser.add_argument("--alignment-surrogate", default="reward_ber_delta")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    pretrained = Path(args.pretrained)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_config = _load_model_config(config, pretrained)
    base_model = _build_equalizer(model_config, pretrained, args.device)

    modulation_rows = []
    peft_rows = []
    window_peft_rows = []
    for delay in args.delays:
        for snr_db in args.snrs:
            for pilot_total in args.pilot_totals:
                for layout in args.pilot_layouts:
                    for seed in args.seeds:
                        env = CommunicationEnvironment(
                            CommEnvConfig(
                                level="B",
                                max_delay=int(delay),
                                snr_db=float(snr_db),
                                total_pilot=int(pilot_total),
                                layout=str(layout),
                                seed=90_000 + int(seed),
                            )
                        )
                        start = env.reset_episode()
                        cir = _estimate_cir_from_known_frame(start.acquisition, int(delay)).to(args.device)
                        soft_tail = start.initial_soft_tail.to(args.device)
                        candidates = structured_modulation_candidates(num_blocks=len(base_model.blocks))
                        window_frames = []
                        window_start_tail = soft_tail.detach().clone()
                        window_index = 0
                        peft_candidates = [
                            {"name": "identity", "groups": set(), "lr": 0.0, "steps": 0},
                            *_focused_peft_candidates(args.peft_lr, args.peft_steps),
                        ]
                        for frame_index in range(1, int(args.frames) + 1):
                            frame = env.next_frame()
                            condition = condition_from_cir(cir, float(snr_db))
                            window_frames.append(frame)
                            rows = evaluate_modulation_candidates(
                                model=base_model,
                                frame=frame,
                                condition=condition,
                                soft_tail=soft_tail,
                                candidates=candidates,
                            )
                            for row in rows:
                                row.update(
                                    {
                                        "level": "B",
                                        "delay": int(delay),
                                        "snr_db": float(snr_db),
                                        "pilot_total": int(pilot_total),
                                        "pilot_layout": str(layout),
                                        "seed": int(seed),
                                        "frame": int(frame_index),
                                    }
                                )
                            modulation_rows.extend(rows)

                            identity_row = _identity_row_from_modulation(rows)
                            identity_row.update(
                                {
                                    "method": "focused_peft",
                                    "action_space": "focused_peft",
                                    "updated_groups": ["identity"],
                                    "adapt_steps": 0,
                                    "peft_lr": 0.0,
                                }
                            )
                            peft_rows.append(identity_row)
                            for candidate in _focused_peft_candidates(args.peft_lr, args.peft_steps):
                                peft_model = copy.deepcopy(base_model)
                                peft_result = apply_adapt_only_peft_update(
                                    model=peft_model,
                                    frame=frame,
                                    condition=condition,
                                    soft_tail=soft_tail,
                                    groups=set(candidate["groups"]),
                                    lr=float(candidate["lr"]),
                                    steps=int(candidate["steps"]),
                                )
                                peft_result.update(
                                    {
                                        "action_name": str(candidate["name"]),
                                        "action_space": "focused_peft",
                                        "peft_lr": float(candidate["lr"]),
                                        "level": "B",
                                        "delay": int(delay),
                                        "snr_db": float(snr_db),
                                        "pilot_total": int(pilot_total),
                                        "pilot_layout": str(layout),
                                        "seed": int(seed),
                                        "frame": int(frame_index),
                                    }
                                )
                                peft_rows.append(peft_result)

                            soft_tail = _next_identity_tail(base_model, frame, condition, soft_tail, args.device)
                            if len(window_frames) >= max(1, int(args.window_size)):
                                window_rows = evaluate_peft_window_candidates(
                                    model=base_model,
                                    frames=window_frames,
                                    condition=condition,
                                    soft_tail=window_start_tail,
                                    candidates=peft_candidates,
                                    window_index=window_index,
                                )
                                for row in window_rows:
                                    row.update(
                                        {
                                            "level": "B",
                                            "delay": int(delay),
                                            "snr_db": float(snr_db),
                                            "pilot_total": int(pilot_total),
                                            "pilot_layout": str(layout),
                                            "seed": int(seed),
                                        }
                                    )
                                window_peft_rows.extend(window_rows)
                                window_frames = []
                                window_start_tail = soft_tail.detach().clone()
                                window_index += 1
                        if window_frames:
                            window_rows = evaluate_peft_window_candidates(
                                model=base_model,
                                frames=window_frames,
                                condition=condition_from_cir(cir, float(snr_db)),
                                soft_tail=window_start_tail,
                                candidates=peft_candidates,
                                window_index=window_index,
                            )
                            for row in window_rows:
                                row.update(
                                    {
                                        "level": "B",
                                        "delay": int(delay),
                                        "snr_db": float(snr_db),
                                        "pilot_total": int(pilot_total),
                                        "pilot_layout": str(layout),
                                        "seed": int(seed),
                                    }
                                )
                            window_peft_rows.extend(window_rows)

    alignment_source = peft_rows if args.action_space == "focused_peft" else modulation_rows
    alignment_rows = [row for row in alignment_source if row["action_name"] != "identity"]
    reward_alignment = summarize_reward_data_correlation(
        alignment_rows,
        threshold=0.6,
        surrogate_name=str(args.alignment_surrogate),
    )
    reward_surrogates = summarize_reward_surrogates(
        alignment_rows,
        threshold=0.6,
    )
    reward_selected_actions = summarize_reward_selected_actions(alignment_source, surrogate_name=str(args.alignment_surrogate))
    window_alignment_source = window_peft_rows if args.action_space == "focused_peft" else alignment_source
    windowed_reward_alignment = summarize_windowed_reward_data_correlation(
        [row for row in window_alignment_source if row["action_name"] != "identity"],
        window_size=int(args.window_size),
        threshold=0.6,
        surrogate_name=str(args.alignment_surrogate),
    )
    windowed_reward_selected_actions = summarize_windowed_selected_actions(
        window_alignment_source,
        window_size=int(args.window_size),
        surrogate_name=str(args.alignment_surrogate),
    )
    offline_reference = _summarize_offline_nn(modulation_rows)
    action_summary = _summarize_action_space(modulation_rows)
    peft_summary = _summarize_peft(peft_rows)
    difficulty = run_level_b_difficulty_scan(
        delays=args.delays,
        snrs=args.snrs,
        pilot_totals=args.pilot_totals,
        pilot_layouts=args.pilot_layouts,
        seeds=args.seeds,
        frames=args.frames,
        output_dir=output_dir,
        seed_offset=90_000,
    )
    payload = {
        "schema_version": "research-diagnostics-v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "config": {
            "delays": args.delays,
            "snrs": args.snrs,
            "pilot_totals": args.pilot_totals,
            "pilot_layouts": args.pilot_layouts,
            "seeds": args.seeds,
            "frames": int(args.frames),
            "device": str(args.device),
            "pretrained": str(pretrained),
            "action_space": str(args.action_space),
            "alignment_surrogate": str(args.alignment_surrogate),
            "window_size": int(args.window_size),
        },
        "offline_nn_reference": offline_reference,
        "reward_data_alignment": reward_alignment,
        "reward_surrogates": reward_surrogates,
        "reward_selected_actions": reward_selected_actions,
        "windowed_reward_data_alignment": windowed_reward_alignment,
        "windowed_reward_selected_actions": windowed_reward_selected_actions,
        "modulation_action_space": action_summary,
        "peft_vs_modulation": peft_summary,
        "level_b_difficulty": {
            "summary": difficulty["summary"],
            "rows_count": len(difficulty["rows"]),
        },
        "raw_files": {
            "modulation_rows": "modulation_action_rows.jsonl",
            "peft_rows": "peft_rows.jsonl",
            "window_peft_rows": "window_peft_rows.jsonl",
            "difficulty": "level_b_difficulty.json",
        },
    }
    _write_jsonl(output_dir / "modulation_action_rows.jsonl", modulation_rows)
    _write_jsonl(output_dir / "peft_rows.jsonl", peft_rows)
    _write_jsonl(output_dir / "window_peft_rows.jsonl", window_peft_rows)
    (output_dir / "research_diagnostics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output_dir}")


def _focused_peft_candidates(base_lr: float, base_steps: int) -> list[dict]:
    """返回收缩后的诊断动作空间：只保留真实 PEFT 更新动作。"""

    steps = max(1, int(base_steps))
    lr = float(base_lr)
    return [
        {"name": "peft_head_light", "groups": {"head"}, "lr": lr, "steps": steps},
        {"name": "peft_head_fast", "groups": {"head"}, "lr": 5e-4, "steps": steps},
        {"name": "peft_adapter_lora_conservative", "groups": {"adapter", "attention_lora", "ffn_lora"}, "lr": 5e-5, "steps": steps},
        {"name": "peft_adapter_lora_light", "groups": {"adapter", "attention_lora", "ffn_lora"}, "lr": lr, "steps": steps},
        {"name": "peft_adapter_lora_head_light", "groups": {"adapter_lora"}, "lr": lr, "steps": steps},
    ]


def _identity_row_from_modulation(rows: list[dict]) -> dict:
    """从低维 modulation 诊断中取 identity 行，作为 focused PEFT no-op 基线。"""

    identity = next(row for row in rows if row["action_name"] == "identity")
    return dict(identity)


@torch.no_grad()
def _next_identity_tail(model, frame, condition, soft_tail: torch.Tensor, device: str) -> torch.Tensor:
    from evaluation.research_diagnostics import _condition_to_device, _frame_to_device

    model.eval()
    frame = _frame_to_device(frame, device)
    condition = _condition_to_device(condition, device)
    tail = soft_tail.to(device).unsqueeze(0).to(torch.complex64)
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame.model_region_ids.unsqueeze(0).long()
    logits, _ = model(rx_iq, condition, region_ids, tail)
    tail_len = soft_tail.numel()
    next_tail_real = torch.tanh(logits.squeeze(0)[-tail_len:] / 2.0)
    return torch.complex(next_tail_real, torch.zeros_like(next_tail_real)).detach()


def _summarize_action_space(rows: list[dict]) -> dict:
    non_identity = [row for row in rows if row["action_name"] != "identity"]
    grouped = defaultdict(list)
    for row in non_identity:
        grouped[row["action_name"]].append(row)
    per_action = []
    for action, values in sorted(grouped.items()):
        reward_values = [float(row["reward_loss_improvement"]) for row in values]
        data_values = [float(row["data_ber_improvement"]) for row in values]
        per_action.append(
            {
                "action_name": action,
                "mean_reward_loss_improvement": _mean(reward_values),
                "mean_data_ber_improvement": _mean(data_values),
                "fraction_reward_improved": _fraction_positive(reward_values),
                "fraction_data_improved": _fraction_positive(data_values),
                "samples": len(values),
            }
        )
    return {
        "metric": "low_dim_modulation_action_effectiveness",
        "num_action_samples": len(non_identity),
        "fraction_reward_improved": _fraction_positive([float(row["reward_loss_improvement"]) for row in non_identity]),
        "fraction_data_improved": _fraction_positive([float(row["data_ber_improvement"]) for row in non_identity]),
        "fraction_both_improved": _fraction_both(non_identity),
        "best_mean_data_action": max(per_action, key=lambda row: row["mean_data_ber_improvement"]) if per_action else None,
        "per_action": per_action,
    }


def _summarize_offline_nn(rows: list[dict]) -> dict:
    identity_rows = [row for row in rows if row["action_name"] == "identity"]
    grouped = defaultdict(list)
    for row in identity_rows:
        grouped[(row["delay"], row["snr_db"], row["pilot_total"], row["pilot_layout"])].append(float(row["data_ber_before"]))
    per_config = [
        {
            "delay": key[0],
            "snr_db": key[1],
            "pilot_total": key[2],
            "pilot_layout": key[3],
            "mean_ber_data": _mean(values),
            "frames": len(values),
        }
        for key, values in sorted(grouped.items())
    ]
    return {
        "metric": "offline_nn_reference",
        "mean_ber_data": _mean([float(row["data_ber_before"]) for row in identity_rows]),
        "per_config": per_config,
    }


def _summarize_peft(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped["+".join(row["updated_groups"])].append(row)
    per_group = []
    for group, values in sorted(grouped.items()):
        per_group.append(
            {
                "groups": group,
                "mean_reward_loss_improvement": _mean([float(row["reward_loss_improvement"]) for row in values]),
                "mean_data_ber_improvement": _mean([float(row["data_ber_improvement"]) for row in values]),
                "fraction_reward_improved": _fraction_positive([float(row["reward_loss_improvement"]) for row in values]),
                "fraction_data_improved": _fraction_positive([float(row["data_ber_improvement"]) for row in values]),
                "samples": len(values),
            }
        )
    return {
        "metric": "adapt_only_peft_vs_low_dim_modulation",
        "peft_groups": per_group,
        "diagnostic_uses_data_labels": True,
        "online_update_uses_data_labels": False,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _fraction_positive(values: list[float]) -> float:
    return float(sum(1 for value in values if value > 0.0) / max(1, len(values)))


def _fraction_both(rows: list[dict]) -> float:
    return float(
        sum(
            1
            for row in rows
            if float(row["reward_loss_improvement"]) > 0.0 and float(row["data_ber_improvement"]) > 0.0
        )
        / max(1, len(rows))
    )


if __name__ == "__main__":
    main()
