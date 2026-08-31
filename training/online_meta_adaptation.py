# -*- coding: utf-8 -*-
"""EME 序列化 Pilot 在线元适配与安全更新。

离线训练使用 Adapt Pilot 产生内循环更新，再用留出的 Reward Pilot
计算一阶外层目标。在线运行只执行内循环，并用 Reward Pilot 做回滚门控。
Data 区域从不参与在线更新。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition
from agent.unfolded_equalizer import UnfoldedEqualizer


@dataclass(frozen=True)
class PilotMetaStepResult:
    """一帧 Pilot 元适配的可审计结果。"""

    accepted: bool
    reward_pilot_count: int
    adapt_pilot_count: int
    pre_adapt_reward_loss: float
    post_adapt_reward_loss: float
    adapt_loss: float
    meta_loss: float
    parameter_delta_norm: float
    outer_target: str
    data_labels_used_online: bool
    updated_tail: torch.Tensor | None = None


def pilot_meta_train_step(
    model: UnfoldedEqualizer,
    frame,
    condition: CIRCondition,
    soft_tail: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    groups: Iterable[str] = ("head", "conditioner_film"),
    inner_steps: int = 1,
    inner_learning_rate: float = 1e-4,
    max_delta_norm: float = 0.5,
) -> PilotMetaStepResult:
    """执行一阶 Pilot 元训练步并更新模型的基础参数。

    内循环只读取 Adapt Pilot 的已知符号。外层目标只读取 Reward Pilot
    的已知符号；Data 标签不进入这个函数的计算图。
    """

    tensors = _prepare_frame(frame, condition, soft_tail, model)
    adapt_mask = tensors["adapt_mask"]
    reward_mask = tensors["reward_mask"]
    if int(adapt_mask.sum()) == 0:
        raise ValueError("元适配要求至少存在一个 Adapt Pilot。")
    if int(reward_mask.sum()) == 0:
        raise ValueError("元适配外层目标要求至少存在一个 Reward Pilot。")
    if inner_learning_rate <= 0.0:
        raise ValueError("inner_learning_rate 必须为正数。")

    group_set = set(groups)
    snapshot = model.peft.snapshot(group_set)
    requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    was_training = model.training
    model.eval()
    model.set_trainable_groups(group_set)
    inner_parameters = model.trainable_parameters()
    if not inner_parameters:
        raise ValueError("指定的 PEFT group 没有可训练参数。")

    preserve_outer_update = False
    try:
        with torch.no_grad():
            before_logits = _forward(model, tensors)
            pre_reward = _masked_bce(before_logits, tensors["bits"], reward_mask)

        inner_optimizer = torch.optim.SGD(inner_parameters, lr=float(inner_learning_rate))
        adapt_loss = torch.zeros((), device=tensors["rx_iq"].device)
        for _ in range(max(1, int(inner_steps))):
            logits = _forward(model, tensors)
            adapt_loss = _masked_bce_tensor(logits, tensors["adapt_targets"], adapt_mask)
            inner_optimizer.zero_grad(set_to_none=True)
            adapt_loss.backward()
            torch.nn.utils.clip_grad_norm_(inner_parameters, 1.0)
            inner_optimizer.step()

        delta_norm = model.peft.delta_norm(snapshot)
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        post_logits = _forward(model, tensors)
        post_reward = _masked_bce_tensor(post_logits, tensors["bits"], reward_mask)
        meta_loss = post_reward
        tail_length = tensors["tail"].shape[-1]
        updated_tail = torch.complex(
            torch.tanh(post_logits.detach().squeeze(0)[-tail_length:] / 2.0),
            torch.zeros(tail_length, device=post_logits.device),
        )
        optimizer.zero_grad(set_to_none=True)
        meta_loss.backward()
        finite = bool(torch.isfinite(meta_loss).item())
        if finite and delta_norm <= float(max_delta_norm):
            model.peft.restore(snapshot)
            optimizer.step()
            accepted = True
            preserve_outer_update = True
        else:
            model.peft.restore(snapshot)
            accepted = False
        return PilotMetaStepResult(
            accepted=accepted,
            reward_pilot_count=int(reward_mask.sum()),
            adapt_pilot_count=int(adapt_mask.sum()),
            pre_adapt_reward_loss=float(pre_reward.detach().cpu()),
            post_adapt_reward_loss=float(post_reward.detach().cpu()),
            adapt_loss=float(adapt_loss.detach().cpu()),
            meta_loss=float(meta_loss.detach().cpu()),
            parameter_delta_norm=float(delta_norm if accepted else 0.0),
            outer_target="reward_pilot",
            data_labels_used_online=False,
            updated_tail=updated_tail,
        )
    finally:
        if not preserve_outer_update:
            model.peft.restore(snapshot)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad_before[name])
        model.train(was_training)


def run_meta_sequence_step(
    model: UnfoldedEqualizer,
    frame,
    condition: CIRCondition,
    soft_tail: torch.Tensor,
    optimizer: torch.optim.Optimizer | None = None,
    groups: Iterable[str] = ("head", "conditioner_film"),
    inner_steps: int = 1,
    inner_learning_rate: float = 1e-4,
    max_delta_norm: float = 0.5,
    reward_guard: Callable[[float, float], bool] | None = None,
) -> tuple[PilotMetaStepResult, torch.Tensor]:
    """执行一帧在线内循环，并按 Reward Pilot 决定是否接受参数更新。

    ``optimizer`` 参数仅为兼容训练入口保留，在线步骤不会使用 Reward
    Pilot 反向更新模型。拒绝时只恢复 PEFT 参数，soft tail 保持输入状态。
    """

    del optimizer
    tensors = _prepare_frame(frame, condition, soft_tail, model)
    adapt_mask = tensors["adapt_mask"]
    reward_mask = tensors["reward_mask"]
    if int(adapt_mask.sum()) == 0 or int(reward_mask.sum()) == 0:
        raise ValueError("在线元适配要求 Adapt Pilot 和 Reward Pilot 均非空。")
    snapshot = model.peft.snapshot(set(groups))
    requires_grad_before = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
    was_training = model.training
    model.eval()
    model.set_trainable_groups(set(groups))
    parameters = model.trainable_parameters()
    try:
        with torch.no_grad():
            before_logits = _forward(model, tensors)
            before = _masked_bce(before_logits, tensors["bits"], reward_mask)
        inner_optimizer = torch.optim.SGD(parameters, lr=float(inner_learning_rate))
        adapt_loss = torch.zeros((), device=tensors["rx_iq"].device)
        for _ in range(max(1, int(inner_steps))):
            logits = _forward(model, tensors)
            adapt_loss = _masked_bce_tensor(logits, tensors["adapt_targets"], adapt_mask)
            inner_optimizer.zero_grad(set_to_none=True)
            adapt_loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            inner_optimizer.step()
        delta_norm = model.peft.delta_norm(snapshot)
        with torch.no_grad():
            after_logits = _forward(model, tensors)
            after = _masked_bce(after_logits, tensors["bits"], reward_mask)
        guard = reward_guard or (lambda old, new: new <= old)
        accepted = bool(torch.isfinite(after).item()) and delta_norm <= float(max_delta_norm) and bool(guard(before, after))
        if not accepted:
            model.peft.restore(snapshot)
            tail = soft_tail.detach().clone()
        else:
            tail_length = soft_tail.numel()
            detected = torch.complex(
                torch.tanh(after_logits.squeeze(0)[-tail_length:] / 2.0),
                torch.zeros(tail_length, device=after_logits.device),
            )
            tail = (0.5 * soft_tail.to(detected.device) + 0.5 * detected).detach()
        result = PilotMetaStepResult(
            accepted=accepted,
            reward_pilot_count=int(reward_mask.sum()),
            adapt_pilot_count=int(adapt_mask.sum()),
            pre_adapt_reward_loss=float(before),
            post_adapt_reward_loss=float(after),
            adapt_loss=float(adapt_loss.detach().cpu()),
            meta_loss=float(after),
            parameter_delta_norm=float(delta_norm if accepted else 0.0),
            outer_target="reward_pilot_guard",
            data_labels_used_online=False,
        )
        return result, tail.to(soft_tail.device)
    finally:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(requires_grad_before[name])
        model.train(was_training)


def _prepare_frame(frame, condition: CIRCondition, soft_tail: torch.Tensor, model: UnfoldedEqualizer) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    view = frame.receiver_view()
    rx = view.rx_symbols.to(device)
    adapt_symbols = view.adapt_symbols.to(device).to(torch.complex64)
    adapt_mask = view.adapt_mask.to(device).bool()
    reward_mask = frame.reward_mask.to(device).bool()
    bits = frame.bits.to(device).float()
    return {
        "rx_iq": torch.stack((rx.real, rx.imag), dim=-1).unsqueeze(0).float(),
        "adapt_symbols": adapt_symbols.unsqueeze(0),
        "adapt_mask": adapt_mask.unsqueeze(0),
        "region_ids": view.model_region_ids.to(device).long().unsqueeze(0),
        "tail": soft_tail.to(device).to(torch.complex64).reshape(1, -1),
        "condition": _condition_to_device(condition, device),
        "reward_mask": reward_mask,
        "bits": bits,
        "adapt_targets": (adapt_symbols.real > 0.0).float().unsqueeze(0),
    }


def _forward(model: UnfoldedEqualizer, tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    logits, _ = model(
        tensors["rx_iq"],
        tensors["condition"],
        tensors["region_ids"],
        tensors["tail"],
        adapt_symbols=tensors["adapt_symbols"],
        adapt_mask=tensors["adapt_mask"],
    )
    return logits


def _masked_bce_tensor(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask if mask.ndim == logits.ndim else mask.unsqueeze(0)
    target_batch = targets if targets.ndim == logits.ndim else targets.unsqueeze(0)
    selected = logits[expanded]
    target = target_batch[expanded]
    if selected.numel() == 0:
        raise ValueError("Pilot 掩码不能为空。")
    return F.binary_cross_entropy_with_logits(selected.float(), target.float())


def _masked_bce(logits: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return _masked_bce_tensor(logits, bits, mask)


def _condition_to_device(condition: CIRCondition, device: torch.device) -> CIRCondition:
    return CIRCondition(
        complex_cir=condition.complex_cir.to(device),
        support_probability=condition.support_probability.to(device),
        noise_variance=condition.noise_variance.to(device),
        confidence=condition.confidence.to(device),
        latent_residual=condition.latent_residual.to(device),
    )


class OnlineMetaTrainer:
    """使用真实通信环境训练可快速在线适配的 EME 模型。"""

    def __init__(self, config: dict, device: torch.device, save_dir: str | None = None):
        self.config = config
        self.device = device
        self.save_dir = save_dir
        from agent.unfolded_equalizer import UnfoldedConfig

        self.model = UnfoldedEqualizer(UnfoldedConfig.from_dict(config["model"])).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=float(config.get("learning_rate", 1e-4))
        )
        self.model.set_trainable_groups({"conditioner_peft"})

    def load_resume(self, checkpoint_path: str | None) -> bool:
        if checkpoint_path is None:
            return False
        from pathlib import Path

        path = Path(checkpoint_path)
        if not path.exists():
            return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = payload.get("state_dict", payload.get("model_state_dict"))
        if state_dict is None:
            raise KeyError("在线元训练 checkpoint 缺少 state_dict。")
        self.model.load_state_dict(state_dict, strict=True)
        return True

    def train(self, steps: int, batch_size: int = 1, smoke: bool = False, resume_loaded: bool = False) -> dict:
        del batch_size, smoke
        from baseline.traditional_equalizers import estimate_phase_residual_vector
        from env.comm_env import CommunicationEnvironment, ReceiverState
        from env.experiment_config import build_comm_env_config
        from training.meta_training import estimate_acquisition_cir_for_profile

        sequence_frames = max(1, int(self.config.get("meta_sequence_frames", 2)))
        groups = set(self.config.get("meta_peft_groups", ["head", "conditioner_film"]))
        inner_steps = max(1, int(self.config.get("meta_inner_steps", 1)))
        inner_lr = float(self.config.get("meta_inner_learning_rate", 1e-4))
        max_delta = float(self.config.get("online_adaptation_max_delta_norm", 0.5))
        state_split = self.config.get("online_meta_state_split", "offline_train")
        total_steps = max(1, int(steps))
        history: list[dict] = []
        for step in range(total_steps):
            delay = int(self.config.get("main_delays", [self.model.config.max_delay])[step % len(self.config.get("main_delays", [self.model.config.max_delay]))])
            snr_values = self.config.get("main_snrs", [10])
            snr_db = float(snr_values[step % len(snr_values)])
            pilot_total = int(self.config.get("pilot_total", 128))
            layout = str(self.config.get("pilot_layout", "prefix"))
            env_config = build_comm_env_config(
                self.config,
                level="B",
                snr_db=snr_db,
                seed=61_000 + step,
                max_delay=delay,
                total_pilot=pilot_total,
                pilot_layout=layout,
                state_split=state_split,
            )
            env = CommunicationEnvironment(env_config)
            start = env.reset_episode()
            cir = estimate_acquisition_cir_for_profile(
                start.acquisition,
                delay,
                str(self.config.get("impairment_profile", "clean")),
            ).to(self.device).to(torch.complex64)
            tail = ReceiverState(start.initial_soft_tail.to(self.device).to(torch.complex64)).soft_tail
            for sequence_index in range(sequence_frames):
                frame = env.next_frame()
                phase_features = estimate_phase_residual_vector(
                    frame.receiver_view(), cir, tail, blocks=4
                )
                from agent.cir_estimator import condition_from_cir

                condition = condition_from_cir(cir, snr_db, phase_features=phase_features)
                result = pilot_meta_train_step(
                    self.model,
                    frame,
                    condition,
                    tail,
                    self.optimizer,
                    groups=groups,
                    inner_steps=inner_steps,
                    inner_learning_rate=inner_lr,
                    max_delta_norm=max_delta,
                )
                if result.updated_tail is not None:
                    tail = result.updated_tail.to(self.device)
                history.append(
                    {
                        "episode": step + 1,
                        "frame": sequence_index + 1,
                        "delay": delay,
                        "snr_db": snr_db,
                        "pre_adapt_reward_loss": result.pre_adapt_reward_loss,
                        "post_adapt_reward_loss": result.post_adapt_reward_loss,
                        "adapt_loss": result.adapt_loss,
                        "meta_loss": result.meta_loss,
                        "adaptation_accepted": result.accepted,
                        "parameter_delta_norm": result.parameter_delta_norm,
                        "data_labels_used_online": False,
                        "condition_source": "acquisition_pilot",
                        "outer_target": result.outer_target,
                        "state_split": state_split,
                    }
                )
        mean_pre = sum(row["pre_adapt_reward_loss"] for row in history) / max(1, len(history))
        mean_post = sum(row["post_adapt_reward_loss"] for row in history) / max(1, len(history))
        return {
            "stage": "online_meta",
            "history": history,
            "condition_source": "acquisition_pilot",
            "sequence_frames": sequence_frames,
            "mean_pre_adapt_reward_loss": mean_pre,
            "mean_post_adapt_reward_loss": mean_post,
            "mean_reward_improvement": mean_pre - mean_post,
            "resume_loaded": bool(resume_loaded),
            "data_labels_used_online": False,
            "state_split": state_split,
        }

    def save(self, save_dir: str, metrics: dict) -> None:
        import json
        from pathlib import Path

        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / "model_config.json").write_text(
            json.dumps(self.model.config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (target / "pretrain_metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        payload = {
            "schema_version": "unfolded-eq-online-meta-v1",
            "model_config": self.model.config.to_dict(),
            "state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        torch.save(payload, target / "model_best.pt")
        torch.save(payload, target / "model_final.pt")
        torch.save(payload, target / "last.pt")
