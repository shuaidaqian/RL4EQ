# -*- coding: utf-8 -*-
"""一阶 episodic meta-training 与固定微调门槛。

本模块只实现离线诊断/训练阶段需要的 Data 查询损失；在线控制器不会从这里
读取 Data 标签或 Data BER。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition, decision_directed_cir_update
from agent.peft import PEFTSnapshot
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
from evaluation.metrics import spearman_reward_data
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from env.frame_structure import Frame, ReceiverFrameView


@dataclass(frozen=True)
class MetaEpisode:
    """单帧 meta episode。

    support 只对应 Adapt Pilot；query 对应 Reward Pilot + Data。Data 标签只在
    trainer 的 query loss 内部使用，不进入 receiver_view。
    """

    receiver_view: ReceiverFrameView
    support_mask: torch.Tensor
    query_mask: torch.Tensor
    target_bits: torch.Tensor
    true_cir: torch.Tensor
    soft_tail: torch.Tensor


@dataclass(frozen=True)
class AdaptedWeights:
    """一阶 inner update 后的 PEFT 快照。"""

    fast_weights: dict[str, torch.Tensor]
    snapshot: PEFTSnapshot
    groups: set[str]
    support_loss: float


@dataclass(frozen=True)
class FixedCandidate:
    groups: set[str]
    steps: int
    iterations: int
    learning_rate: float
    score: float | None = None

    def to_json(self) -> dict:
        payload = asdict(self)
        payload["groups"] = sorted(self.groups)
        return payload


@dataclass(frozen=True)
class FixedGateResult:
    candidate: FixedCandidate
    score: float
    gate_pass: bool
    gate_checked: int


def build_meta_episode(frame: Frame, device: torch.device | None = None) -> MetaEpisode:
    """从离线帧构造 support/query 隔离的 meta episode。"""

    target_device = device or frame.tx_symbols.device
    view = frame.receiver_view()
    receiver_view = ReceiverFrameView(
        rx_symbols=view.rx_symbols.to(target_device),
        adapt_symbols=view.adapt_symbols.to(target_device),
        adapt_mask=view.adapt_mask.to(target_device),
        model_region_ids=view.model_region_ids.to(target_device),
    )
    max_delay = int(frame.tail_symbols.numel()) if frame.tail_symbols is not None else 40
    true_cir = frame.true_cir
    if true_cir is None:
        true_cir = torch.zeros(max_delay + 1, dtype=torch.complex64)
        true_cir[0] = 1.0 + 0.0j
    soft_tail = frame.tail_symbols
    if soft_tail is None:
        soft_tail = torch.zeros(max_delay, dtype=torch.complex64)
    return MetaEpisode(
        receiver_view=receiver_view,
        support_mask=frame.adapt_mask.to(target_device),
        query_mask=(frame.reward_mask | frame.data_mask).to(target_device),
        target_bits=frame.bits.to(target_device).float(),
        true_cir=true_cir.to(target_device),
        soft_tail=soft_tail.to(target_device),
    )


def first_order_inner_update(
    model: UnfoldedEqualizer,
    episode: MetaEpisode,
    groups: Iterable[str],
    steps: int = 1,
    lr: float | None = None,
    learning_rate: float | None = None,
) -> AdaptedWeights:
    """只用 Adapt Pilot 做一阶 PEFT inner update，并恢复原模型参数。"""

    group_set = set(groups)
    actual_lr = float(learning_rate if learning_rate is not None else (lr if lr is not None else 1e-4))
    snapshot = model.peft.snapshot(group_set)
    was_training = model.training
    model.eval()
    model.set_trainable_groups(group_set)
    parameters = model.trainable_parameters()
    if not parameters:
        return AdaptedWeights({}, snapshot, group_set, support_loss=0.0)
    optimizer = torch.optim.SGD(parameters, lr=actual_lr)
    last_loss = torch.zeros((), device=episode.target_bits.device)
    try:
        for _ in range(max(1, steps)):
            optimizer.zero_grad(set_to_none=True)
            logits, _ = _forward_episode(model, episode)
            mask = episode.support_mask.unsqueeze(0)
            last_loss = F.binary_cross_entropy_with_logits(
                logits[mask],
                episode.target_bits.unsqueeze(0)[mask],
            )
            last_loss.backward()
            optimizer.step()
        fast_weights = {
            name: parameter.detach().clone()
            for name, parameter in model.peft.named_group_parameters(group_set)
        }
    finally:
        model.peft.restore(snapshot)
        model.set_trainable_groups(set())
        model.train(was_training)
    return AdaptedWeights(fast_weights, snapshot, group_set, support_loss=float(last_loss.detach().cpu()))


class FixedGate:
    """Best Fixed 搜索器；正式门槛由调用方用逐配置结果决定。"""

    LEGAL_GROUPS = {
        "conditioner_film",
        "adapter",
        "attention_lora",
        "ffn_lora",
        "head",
        "adapter_lora",
        "conditioner_peft",
    }

    def __init__(
        self,
        save_dir: str | Path,
        group_grid: list[set[str]] | None = None,
        steps_grid: list[int] | None = None,
        iterations_grid: list[int] | None = None,
        lr_grid: list[float] | None = None,
    ):
        self.save_dir = Path(save_dir)
        self.group_grid = group_grid or [{"head"}, {"adapter"}, {"adapter_lora"}, {"conditioner_peft"}]
        self.steps_grid = steps_grid or [1, 2, 4]
        self.iterations_grid = iterations_grid or [2, 4, 6, 8]
        self.lr_grid = lr_grid or [1e-4, 3e-4, 1e-3]

    def enumerate_candidates(self) -> list[FixedCandidate]:
        candidates: list[FixedCandidate] = []
        for groups in self.group_grid:
            if not set(groups) <= self.LEGAL_GROUPS:
                raise ValueError(f"非法 PEFT 组：{groups}")
            for steps in self.steps_grid:
                for iterations in self.iterations_grid:
                    for lr in self.lr_grid:
                        candidates.append(FixedCandidate(set(groups), int(steps), int(iterations), float(lr)))
        return candidates

    def select_best(self, scorer: Callable[[FixedCandidate], float], threshold: float = 0.1) -> FixedGateResult:
        scored: list[FixedCandidate] = []
        for candidate in self.enumerate_candidates():
            scored.append(
                FixedCandidate(
                    groups=candidate.groups,
                    steps=candidate.steps,
                    iterations=candidate.iterations,
                    learning_rate=candidate.learning_rate,
                    score=float(scorer(candidate)),
                )
            )
        best = min(scored, key=lambda item: float(item.score))
        result = FixedGateResult(best, float(best.score), bool(float(best.score) < threshold), len(scored))
        self._write_result(result, scored)
        return result

    def _write_result(self, result: FixedGateResult, candidates: list[FixedCandidate]) -> None:
        target = self.save_dir / "fixed_gate"
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "best": result.candidate.to_json(),
            "gate_pass": result.gate_pass,
            "gate_checked": result.gate_checked,
            "candidates": [candidate.to_json() for candidate in candidates],
        }
        (target / "best_fixed.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class MetaTrainer:
    """Smoke 级 first-order meta trainer，保持正式接口可扩展。"""

    def __init__(self, config: dict, device: torch.device, save_dir: str | Path):
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.model = UnfoldedEqualizer(UnfoldedConfig.from_dict(config["model"])).to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)

    def load_resume(self, checkpoint_path: str | Path | None) -> bool:
        if checkpoint_path is None:
            return False
        path = Path(checkpoint_path)
        if not path.exists():
            return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["state_dict"], strict=True)
        return True

    def train(self, steps: int, batch_size: int, smoke: bool = False, resume_loaded: bool = False) -> dict:
        del batch_size
        history = []
        from env.frame_structure import FrameConfig, FrameGenerator

        frame_len = self.model.config.frame_len
        max_delay = self.model.config.max_delay
        total_pilot = 64 if frame_len <= 128 else int(self.config.get("pilot_total", 128))
        frame_generator = FrameGenerator(
            FrameConfig(frame_len=frame_len, total_pilot=total_pilot, layout=self.config.get("pilot_layout", "prefix"), max_delay=max_delay),
            seed=31,
        )
        for step in range(max(1, steps)):
            frame = frame_generator.generate(step)
            cir = torch.zeros(max_delay + 1, dtype=torch.complex64)
            cir[0] = 1.0 + 0.0j
            tail = torch.zeros(max_delay, dtype=torch.complex64)
            frame = frame.with_channel_output(frame.tx_symbols.clone(), tail, cir)
            episode = build_meta_episode(frame, self.device)
            first_order_inner_update(self.model, episode, groups={"head"}, steps=1, learning_rate=1e-4)
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            self.optimizer.zero_grad(set_to_none=True)
            logits, _ = _forward_episode(self.model, episode)
            query_mask = episode.query_mask.unsqueeze(0)
            loss = F.binary_cross_entropy_with_logits(logits[query_mask], episode.target_bits.unsqueeze(0)[query_mask])
            loss.backward()
            self.optimizer.step()
            history.append({"step": step + 1, "query_loss": float(loss.detach().cpu())})
        gate = FixedGate(self.save_dir, group_grid=[{"head"}] if smoke else None)
        gate_result = gate.select_best(lambda candidate: 0.5 + candidate.steps * 0.001, threshold=0.1)
        return {
            "stage": "meta",
            "history": history,
            "gate_pass": gate_result.gate_pass,
            "gate_checked": gate_result.gate_checked,
            "gate_smoke": smoke,
            "resume_loaded": resume_loaded,
            "note": "smoke 训练不强制 Best Fixed < 0.1；正式门槛由开发规模评估执行。",
        }

    def save(self, save_dir: str | Path, metrics: dict) -> None:
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_config.json").write_text(
            json.dumps(self.model.config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (target / "pretrain_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        payload = {
            "schema_version": "unfolded-eq-v1",
            "model_config": self.model.config.to_dict(),
            "state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        torch.save(payload, target / "model_best.pt")
        torch.save(payload, target / "model_final.pt")
        torch.save(payload, target / "last.pt")


def evaluate_best_fixed_level_b(
    config: dict,
    output_dir: str | Path,
    frames_per_config: int = 5,
    seeds: list[int] | None = None,
    cg_grid: list[int] | None = None,
    refine_grid: list[int] | None = None,
) -> dict:
    """用 acquisition-CIR 固定检测器完成 Best Fixed gate。

    候选选择只看 Reward Pilot BER；Data BER 仅用于仿真 gate 报告。
    """

    seed_list = seeds or [0, 1, 2, 3, 4]
    candidate_grid = [
        {"cg_iterations": int(cg), "refine_iterations": int(refine)}
        for cg in (cg_grid or [32, 64])
        for refine in (refine_grid or [1, 2])
    ]
    scored = []
    for candidate in candidate_grid:
        reward_bers = []
        data_bers = []
        per_config = []
        for delay in config.get("main_delays", [20, 30, 40]):
            for snr_db in config.get("main_snrs", [0, 5, 10, 15]):
                config_reward = []
                config_data = []
                for seed in seed_list:
                    env = CommunicationEnvironment(
                        CommEnvConfig(
                            level="B",
                            max_delay=int(delay),
                            snr_db=float(snr_db),
                            rho=float(config.get("rho", 0.99)),
                            total_pilot=int(config.get("pilot_total", 128)),
                            layout=str(config.get("pilot_layout", "prefix")),
                            seed=20_000 + int(seed),
                            impairment_profile=str(config.get("impairment_profile", "clean")),
                        )
                    )
                    start = env.reset_episode()
                    receiver_state = ReceiverState(start.initial_soft_tail)
                    cir = estimate_acquisition_cir_for_profile(
                        start.acquisition,
                        int(delay),
                        str(config.get("impairment_profile", "clean")),
                    )
                    for _ in range(frames_per_config):
                        frame = env.next_frame()
                        result = perfect_csi_bpsk_refine_detect(
                            frame.rx_symbols,
                            cir,
                            receiver_state.soft_tail,
                            torch.tensor(10.0 ** (-float(snr_db) / 10.0)),
                            cg_iterations=candidate["cg_iterations"],
                            refine_iterations=candidate["refine_iterations"],
                        )
                        receiver_state.update_tail(result.soft_tail)
                        cir = decision_directed_cir_update(frame, result.logits, int(delay), cir, alpha=0.2)
                        reward_ber = bit_error_rate(result.logits[frame.reward_mask], frame.bits[frame.reward_mask])
                        data_ber = bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask])
                        config_reward.append(reward_ber)
                        config_data.append(data_ber)
                        reward_bers.append(reward_ber)
                        data_bers.append(data_ber)
                per_config.append(
                    {
                        "level": "B",
                        "delay": int(delay),
                        "snr_db": float(snr_db),
                        "ber_reward_pilot": float(sum(config_reward) / max(1, len(config_reward))),
                        "ber_data": float(sum(config_data) / max(1, len(config_data))),
                        "frames": frames_per_config,
                        "seeds": list(seed_list),
                    }
                )
        scored.append(
            {
                "candidate": candidate,
                "mean_reward_pilot_ber": float(sum(reward_bers) / max(1, len(reward_bers))),
                "mean_ber_data": float(sum(data_bers) / max(1, len(data_bers))),
                "per_config": per_config,
            }
        )
    best = min(
        scored,
        key=lambda item: (
            item["mean_reward_pilot_ber"],
            item["candidate"]["cg_iterations"],
            item["candidate"]["refine_iterations"],
        ),
    )
    threshold = 0.1
    result = {
        "gate": "best_fixed_acquisition_cir",
        "gate_threshold": threshold,
        "gate_pass": all(row["ber_data"] < threshold for row in best["per_config"]),
        "selected": {
            **best["candidate"],
            "selection_metric": "mean_reward_pilot_ber",
            "mean_reward_pilot_ber": best["mean_reward_pilot_ber"],
            "mean_ber_data": best["mean_ber_data"],
        },
        "per_config": best["per_config"],
        "candidates": [
            {
                "candidate": item["candidate"],
                "mean_reward_pilot_ber": item["mean_reward_pilot_ber"],
                "mean_ber_data": item["mean_ber_data"],
            }
            for item in scored
        ],
    }
    target = Path(output_dir) / "fixed_gate"
    target.mkdir(parents=True, exist_ok=True)
    (target / "best_fixed.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def evaluate_reward_data_alignment(
    config: dict,
    output_dir: str | Path,
    frames_per_config: int = 5,
    seeds: list[int] | None = None,
    pilot_total: int | None = None,
    pilot_layout: str | None = None,
    action_grid: list[tuple[int, int]] | None = None,
) -> dict:
    """验证 Reward Pilot 改善与 Data BER 改善的真实配对 Spearman。

    单帧 BER 离散性较强，因此 gate 采用配置×动作候选分组后的平均改善。
    """

    seed_list = seeds or [0, 1, 2, 3, 4]
    actions = action_grid or [(16, 0), (16, 1), (32, 1), (32, 2)]
    grouped_pairs = []
    for delay in config.get("main_delays", [20, 30, 40]):
        for snr_db in config.get("main_snrs", [0, 5, 10, 15]):
            for cg_iterations, refine_iterations in actions:
                reward_improvements = []
                data_improvements = []
                for seed in seed_list:
                    env = CommunicationEnvironment(
                        CommEnvConfig(
                            level="B",
                            max_delay=int(delay),
                            snr_db=float(snr_db),
                            rho=float(config.get("rho", 0.99)),
                            total_pilot=int(pilot_total or config.get("pilot_total", 128)),
                            layout=str(pilot_layout or config.get("pilot_layout", "prefix")),
                            seed=30_000 + int(seed),
                            impairment_profile=str(config.get("impairment_profile", "clean")),
                        )
                    )
                    start = env.reset_episode()
                    receiver_state = ReceiverState(start.initial_soft_tail)
                    cir = estimate_acquisition_cir_for_profile(
                        start.acquisition,
                        int(delay),
                        str(config.get("impairment_profile", "clean")),
                    )
                    for _ in range(frames_per_config):
                        frame = env.next_frame()
                        sigma = torch.tensor(10.0 ** (-float(snr_db) / 10.0))
                        before = perfect_csi_bpsk_refine_detect(
                            frame.rx_symbols,
                            cir,
                            receiver_state.soft_tail,
                            sigma,
                            cg_iterations=8,
                            refine_iterations=0,
                        )
                        after = perfect_csi_bpsk_refine_detect(
                            frame.rx_symbols,
                            cir,
                            receiver_state.soft_tail,
                            sigma,
                            cg_iterations=int(cg_iterations),
                            refine_iterations=int(refine_iterations),
                        )
                        receiver_state.update_tail(after.soft_tail)
                        reward_improvements.append(
                            _masked_bce(before.logits, frame.bits, frame.reward_mask)
                            - _masked_bce(after.logits, frame.bits, frame.reward_mask)
                        )
                        data_improvements.append(
                            bit_error_rate(before.logits[frame.data_mask], frame.bits[frame.data_mask])
                            - bit_error_rate(after.logits[frame.data_mask], frame.bits[frame.data_mask])
                        )
                grouped_pairs.append(
                    {
                        "delay": int(delay),
                        "snr_db": float(snr_db),
                        "cg_iterations": int(cg_iterations),
                        "refine_iterations": int(refine_iterations),
                        "reward_improvement": float(sum(reward_improvements) / max(1, len(reward_improvements))),
                        "data_ber_improvement": float(sum(data_improvements) / max(1, len(data_improvements))),
                    }
                )
    spearman = spearman_reward_data(
        [pair["reward_improvement"] for pair in grouped_pairs],
        [pair["data_ber_improvement"] for pair in grouped_pairs],
    )
    threshold = 0.6
    result = {
        "gate": "reward_data_spearman",
        "gate_threshold": threshold,
        "gate_pass": bool(spearman.correlation >= threshold),
        "spearman": float(spearman.correlation),
        "num_pairs": spearman.n,
        "pairing": "grouped_by_config_and_action",
        "pilot_total": int(pilot_total or config.get("pilot_total", 128)),
        "pilot_layout": str(pilot_layout or config.get("pilot_layout", "prefix")),
        "pairs": grouped_pairs,
    }
    target = Path(output_dir) / "reward_alignment"
    target.mkdir(parents=True, exist_ok=True)
    (target / "alignment.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _estimate_cir_from_known_frame(frame: Frame, max_delay: int) -> torch.Tensor:
    rows = []
    targets = []
    tx = frame.tx_symbols.to(torch.complex64)
    rx = frame.rx_symbols.to(torch.complex64)
    for pos in range(max_delay, tx.numel()):
        row = torch.zeros(max_delay + 1, dtype=torch.complex64)
        for delay in range(max_delay + 1):
            row[delay] = tx[pos - delay]
        rows.append(row)
        targets.append(rx[pos])
    design = torch.stack(rows, dim=0)
    target = torch.stack(targets, dim=0)
    cir = torch.linalg.lstsq(design, target).solution.to(torch.complex64)
    return cir / torch.sqrt(torch.sum(torch.abs(cir) ** 2).clamp_min(1e-12))


def estimate_acquisition_cir_for_profile(frame: Frame, max_delay: int, impairment_profile: str = "clean") -> torch.Tensor:
    """按实验 profile 选择传统可见的 acquisition CIR 估计方式。"""

    if str(impairment_profile) == "clean":
        return _estimate_cir_from_known_frame(frame, int(max_delay))
    from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo

    cir, _ = estimate_acquisition_cir_with_cfo(frame, int(max_delay))
    return cir


def _masked_bce(logits: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor) -> float:
    if int(mask.sum().item()) == 0:
        return 0.0
    return float(F.binary_cross_entropy_with_logits(logits[mask].float(), bits[mask].float()).item())


def _forward_episode(model: UnfoldedEqualizer, episode: MetaEpisode) -> tuple[torch.Tensor, torch.Tensor]:
    rx = episode.receiver_view.rx_symbols.unsqueeze(0)
    rx_iq = torch.stack((rx.real, rx.imag), dim=-1).to(torch.float32)
    cir = episode.true_cir.unsqueeze(0).to(torch.complex64)
    condition = CIRCondition(
        complex_cir=cir,
        support_probability=(cir.abs() > 0).float(),
        noise_variance=torch.full((1,), 0.01, device=rx_iq.device),
        confidence=torch.ones(1, device=rx_iq.device),
        latent_residual=torch.zeros(1, 96, device=rx_iq.device),
    )
    return model(
        rx_iq,
        condition,
        episode.receiver_view.model_region_ids.unsqueeze(0).long(),
        episode.soft_tail.unsqueeze(0).to(torch.complex64),
        adapt_symbols=episode.receiver_view.adapt_symbols.unsqueeze(0).to(torch.complex64),
        adapt_mask=episode.receiver_view.adapt_mask.unsqueeze(0).bool(),
    )
