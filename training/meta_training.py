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

from agent.cir_estimator import CIRCondition
from agent.peft import PEFTSnapshot
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
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
            FrameConfig(frame_len=frame_len, total_pilot=total_pilot, layout=self.config.get("pilot_layout", "multi_block"), max_delay=max_delay),
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
    )
