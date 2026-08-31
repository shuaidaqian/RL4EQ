# -*- coding: utf-8 -*-
"""Pilot 条件 Level A 到 Level B 课程监督训练。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition, decision_directed_cir_update
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate, perfect_csi_bpsk_refine_detect
from baseline.traditional_equalizers import estimate_phase_residual_vector
from env.comm_env import CommunicationEnvironment, ReceiverState
from env.experiment_config import build_comm_env_config, effective_channel_metadata


@dataclass(frozen=True)
class CurriculumPhase:
    name: str
    level: str
    total_pilot: int
    layout: str
    snr_db: int
    max_delay: int
    uses_pilot_condition: bool = True
    delay_grid: tuple[int, ...] | None = None
    snr_grid: tuple[float, ...] | None = None

    def sample_delay_snr(self, sample_index: int) -> tuple[int, float]:
        """按样本序号循环覆盖该阶段的 delay/SNR 训练网格。"""

        delays = self.delay_grid or (int(self.max_delay),)
        snrs = self.snr_grid or (float(self.snr_db),)
        pairs = [(int(delay), float(snr)) for delay in delays for snr in snrs]
        return pairs[int(sample_index) % len(pairs)]


@dataclass(frozen=True)
class TrainingCIRSelection:
    """一次训练样本使用的 CIR 条件及其来源。"""

    cir: torch.Tensor
    source: str


def load_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _pad_complex_1d(value: torch.Tensor, target_len: int) -> torch.Tensor:
    """把一维复向量右侧补零到目标长度；超长时截断。"""

    vector = value.reshape(-1).to(torch.complex64)
    target = int(target_len)
    if vector.numel() == target:
        return vector
    if vector.numel() > target:
        return vector[:target]
    return F.pad(vector, (0, target - vector.numel()))


def select_training_tail(
    *,
    frame_tail: torch.Tensor,
    receiver_tail: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """选择训练帧的跨帧 ISI 状态。"""

    selected_mode = str(mode)
    if selected_mode == "oracle":
        return frame_tail
    if selected_mode == "online_soft":
        return receiver_tail
    if selected_mode == "corrupted_oracle":
        return frame_tail
    raise ValueError(f"未知 training_tail_mode：{selected_mode}")


def corrupt_training_tail(
    tail: torch.Tensor,
    *,
    flip_probability: float,
    sample_index: int,
) -> torch.Tensor:
    """用可复现的符号翻转模拟不完全可靠的历史帧判决。"""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(83_000 + int(sample_index) * 1_009)
    random_values = torch.rand(tail.shape, generator=generator, dtype=torch.float32).to(tail.device)
    flip_mask = random_values < float(flip_probability)
    signs = torch.where(
        flip_mask,
        torch.full_like(random_values, -1.0),
        torch.ones_like(random_values),
    )
    return tail.to(torch.complex64) * signs.to(torch.complex64)


def select_validation_tail(
    *,
    frame_tail: torch.Tensor,
    receiver_tail: torch.Tensor,
    sequence_state: bool,
) -> torch.Tensor:
    """让离线序列验证的 phase 特征与实际输入 tail 保持一致。"""

    return receiver_tail if bool(sequence_state) else frame_tail


def _phase_feature_latent(phase_features: list[torch.Tensor], device: torch.device) -> torch.Tensor:
    """构造 phase-aware conditioner latent。"""

    batch_size = len(phase_features)
    latent = torch.zeros(int(batch_size), 96, device=device, dtype=torch.float32)
    if batch_size == 0:
        return latent
    features = torch.stack([item.reshape(-1).to(torch.float32) for item in phase_features], dim=0).to(device)
    count = min(latent.shape[1], features.shape[1])
    latent[:, :count] = features[:, :count]
    return latent


def _normalize_cir(cir: torch.Tensor) -> torch.Tensor:
    """把 CIR 归一化到单位功率，避免 augmentation 改变 SNR 定义。"""

    vector = cir.reshape(-1).to(torch.complex64)
    power = torch.sum(torch.abs(vector) ** 2).clamp_min(1e-12)
    return vector / torch.sqrt(power)


def _estimate_cir_from_tx_symbols(rx_symbols: torch.Tensor, tx_symbols: torch.Tensor, max_delay: int) -> torch.Tensor:
    """用给定发送符号序列做最小二乘 CIR 估计。"""

    rx = rx_symbols.reshape(-1).to(torch.complex64)
    tx = tx_symbols.reshape(-1).to(torch.complex64)
    rows = []
    targets = []
    for pos in range(int(max_delay), int(tx.numel())):
        row = torch.zeros(int(max_delay) + 1, dtype=torch.complex64, device=tx.device)
        for delay in range(int(max_delay) + 1):
            row[delay] = tx[pos - delay]
        rows.append(row)
        targets.append(rx[pos])
    if not rows:
        return torch.zeros(int(max_delay) + 1, dtype=torch.complex64, device=tx.device)
    estimate = torch.linalg.lstsq(torch.stack(rows), torch.stack(targets)).solution.to(torch.complex64)
    return _normalize_cir(estimate)


def supervised_equalization_loss(
    logits: torch.Tensor,
    bits: torch.Tensor,
    adapt_mask: torch.Tensor,
    reward_mask: torch.Tensor,
    data_mask: torch.Tensor,
    *,
    data_margin_loss_weight: float = 0.0,
    data_margin: float = 1.0,
) -> torch.Tensor:
    """离线监督均衡损失。

    基础项保持原研究契约：Data BCE 为主，Adapt/Reward Pilot BCE 各 0.25。
    可选 Data margin 项只使用离线 Data 标签，目的是强化接近判决边界的
    Data 符号；该项不进入在线 observation/reward/PPO。
    """

    data_loss = F.binary_cross_entropy_with_logits(logits[data_mask], bits[data_mask].float())
    adapt_loss = F.binary_cross_entropy_with_logits(logits[adapt_mask], bits[adapt_mask].float())
    reward_loss = F.binary_cross_entropy_with_logits(logits[reward_mask], bits[reward_mask].float())
    loss = data_loss + 0.25 * adapt_loss + 0.25 * reward_loss
    if float(data_margin_loss_weight) > 0.0:
        labels = bits[data_mask].float() * 2.0 - 1.0
        signed_margin = labels * logits[data_mask].float()
        margin_loss = torch.relu(float(data_margin) - signed_margin).mean()
        loss = loss + float(data_margin_loss_weight) * margin_loss
    return loss


def build_curriculum(config: dict[str, Any]) -> list[CurriculumPhase]:
    default_layout = config.get("pilot_layout", "prefix")
    level_b_pilot = int(config.get("pilot_total", 128))
    if config.get("channel_profile") in {"eme_measurement_v1", "eme_long_memory_v2"}:
        delays = tuple(int(value) for value in config.get("main_delays", [int(config["max_delay"])]))
        snrs = tuple(float(value) for value in config.get("main_snrs", [0, 5, 10, 15]))
        return [
            CurriculumPhase(
                "estimated_cir_level_b",
                "B",
                level_b_pilot,
                default_layout,
                int(snrs[0]),
                int(delays[0]),
                delay_grid=delays,
                snr_grid=snrs,
            )
        ]
    return [
        CurriculumPhase("cir_level_a", "A", 160, default_layout, 20, 20),
        CurriculumPhase("perfect_cir_level_a", "A", 128, default_layout, 20, 20),
        CurriculumPhase("estimated_cir_level_a", "A", 96, default_layout, 15, 30),
        CurriculumPhase(
            "estimated_cir_level_b",
            "B",
            level_b_pilot,
            default_layout,
            10,
            40,
            delay_grid=tuple(int(value) for value in config.get("main_delays", [20, 30, 40])),
            snr_grid=tuple(float(value) for value in config.get("main_snrs", [0, 5, 10, 15])),
        ),
    ]


class CurriculumTrainer:
    """执行四阶段 smoke 监督训练并写出 strict-load checkpoint。"""

    def __init__(self, config: dict[str, Any], device: torch.device):
        self.config = config
        model_config = UnfoldedConfig.from_dict(config["model"])
        self.model = UnfoldedEqualizer(model_config).to(device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(config.get("learning_rate", 1e-4)))
        self.device = device
        self.history: list[dict[str, Any]] = []
        self._sample_counter = 0
        self.condition_cir_source_counts: dict[str, int] = {}
        self.best_model_state_dict: dict[str, torch.Tensor] | None = None
        self.best_model_metric: float = float("inf")
        self.best_model_validation: dict[str, Any] | None = None

    def train(self, stage: str, steps: int, batch_size: int, accumulation_steps: int, use_amp: bool) -> dict[str, Any]:
        phases = build_curriculum(self.config)
        selected = phases if stage == "all" else [phase for phase in phases if phase.name == stage]
        if not selected:
            raise ValueError(f"未知训练阶段：{stage}")
        amp_enabled = bool(use_amp and self.device.type == "cuda")
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        best_interval = int(self.config.get("offline_best_validation_interval", 0))
        if best_interval > 0:
            initial_validation = self._validate_offline_nn()
            self._record_best_model(initial_validation, tag="initial")
        for phase in selected:
            for step in range(steps):
                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    loss = self._step_loss(phase, batch_size) / max(1, accumulation_steps)
                scaler.scale(loss).backward()
                if (step + 1) % max(1, accumulation_steps) == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                self.history.append({"phase": phase.name, "step": step + 1, "loss": float(loss.detach().cpu())})
                if best_interval > 0 and (step + 1) % best_interval == 0:
                    validation = self._validate_offline_nn()
                    self._record_best_model(validation, tag=f"{phase.name}:{step + 1}")
                    self.history.append(
                        {
                            "phase": phase.name,
                            "step": step + 1,
                            "validation_mean_ber_data": float(validation["mean_ber_data"]),
                            "best_model_metric": float(self.best_model_metric),
                        }
                    )
        validation = self._validate_level_b()
        offline_validation = self._validate_offline_nn()
        self._record_best_model(offline_validation, tag="final")
        reference_config = build_comm_env_config(
            self.config,
            level="B",
            snr_db=float(self.config.get("main_snrs", [10])[0]),
            seed=int(self.config.get("curriculum_seed", 80_000)),
            max_delay=int(self.config.get("main_delays", [self.model.config.max_delay])[0]),
            total_pilot=int(self.config.get("pilot_total", 128)),
            pilot_layout=str(self.config.get("pilot_layout", "prefix")),
            state_split=self.config.get("offline_state_split"),
        )
        return {
            "history": self.history,
            "condition_cir_sources": dict(self.condition_cir_source_counts),
            "validation": validation,
            "offline_nn_validation": offline_validation,
            "best_offline_nn_validation": self.best_model_validation,
            "best_model_metric": float(self.best_model_metric),
            "effective_channel": effective_channel_metadata(reference_config),
        }

    def save(self, save_dir: str | Path, metrics: dict[str, Any]) -> None:
        target = Path(save_dir)
        target.mkdir(parents=True, exist_ok=True)
        config_path = target / "model_config.json"
        config_path.write_text(json.dumps(self.model.config.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        metrics_path = target / "pretrain_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
        payload = {
            "schema_version": "unfolded-eq-v1",
            "model_config": self.model.config.to_dict(),
            "state_dict": self.model.state_dict(),
            "metrics": metrics,
        }
        best_state = self.best_model_state_dict or self._snapshot_model_state()
        best_payload = dict(payload)
        best_payload["state_dict"] = best_state
        best_payload["selected_by"] = "offline_nn_validation_mean_ber_data"
        best_payload["best_model_metric"] = float(self.best_model_metric)
        best_payload["best_offline_nn_validation"] = self.best_model_validation
        torch.save(best_payload, target / "model_best.pt")
        torch.save(payload, target / "model_final.pt")
        torch.save(payload, target / "last.pt")

    def load_resume(self, checkpoint_path: str | Path | None) -> bool:
        """严格加载 curriculum checkpoint；缺失路径返回 False。"""

        if checkpoint_path is None:
            return False
        path = Path(checkpoint_path)
        if not path.exists():
            return False
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = payload.get("state_dict")
        if state_dict is None:
            raise KeyError("curriculum checkpoint 必须包含 state_dict。")
        self.model.load_state_dict(state_dict, strict=True)
        return True

    def _snapshot_model_state(self) -> dict[str, torch.Tensor]:
        """把当前模型权重复制到 CPU，避免后续训练原地修改 best checkpoint。"""

        return {name: value.detach().cpu().clone() for name, value in self.model.state_dict().items()}

    def _record_best_model(self, validation: dict[str, Any], tag: str) -> None:
        """按离线神经均衡器验证 BER 选择 best checkpoint。"""

        metric = float(validation.get("mean_ber_data", float("inf")))
        if metric < float(self.best_model_metric):
            self.best_model_metric = metric
            self.best_model_validation = dict(validation)
            self.best_model_validation["selected_at"] = str(tag)
            self.best_model_state_dict = self._snapshot_model_state()

    def _step_loss(self, phase: CurriculumPhase, batch_size: int) -> torch.Tensor:
        from training.meta_training import estimate_acquisition_cir_for_profile
        from training.windowed_discrete_ppo import refine_logits_with_tail

        rx_iq_items = []
        bit_items = []
        region_items = []
        tail_items = []
        cir_items = []
        adapt_masks = []
        reward_masks = []
        data_masks = []
        snr_items = []
        phase_feature_items = []
        adapt_symbol_items = []
        base_seed = int(self.config.get("curriculum_seed", 80_000))
        max_frame_offset = int(self.config.get("curriculum_max_frame_offset", 4))
        training_tail_mode = str(self.config.get("training_tail_mode", "oracle"))
        for _ in range(int(batch_size)):
            sample_index = self._sample_counter
            seed = base_seed + sample_index
            max_delay, snr_db = phase.sample_delay_snr(sample_index)
            frame_offset = seed % (max_frame_offset + 1)
            self._sample_counter += 1
            env = CommunicationEnvironment(
                build_comm_env_config(
                    self.config,
                    level=phase.level,
                    snr_db=float(snr_db),
                    seed=seed,
                    max_delay=int(max_delay),
                    total_pilot=int(phase.total_pilot),
                    pilot_layout=str(phase.layout),
                    impairment_profile=str(self.config.get("impairment_profile", "clean")),
                    state_split=self.config.get("offline_state_split"),
                )
            )
            start = env.reset_episode()
            frame = None
            receiver_state = ReceiverState(start.initial_soft_tail.to(self.device).to(torch.complex64))
            impairment_profile = str(self.config.get("impairment_profile", "clean"))
            if impairment_profile == "clean":
                acquisition_cir = estimate_acquisition_cir_for_profile(
                    start.acquisition,
                    int(max_delay),
                    "clean",
                ).to(self.device).to(torch.complex64)
                acquisition_cfo = 0.0
            else:
                from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo

                acquisition_cir, acquisition_cfo = estimate_acquisition_cir_with_cfo(
                    start.acquisition,
                    int(max_delay),
                    cfo_limit=float(self.config.get("acquisition_cfo_limit", 0.004)),
                )
                acquisition_cir = acquisition_cir.to(self.device).to(torch.complex64)
            for history_index in range(frame_offset + 1):
                candidate_frame = env.next_frame()
                if history_index < frame_offset and training_tail_mode == "online_soft":
                    self._advance_training_receiver_state(
                        candidate_frame,
                        acquisition_cir,
                        float(snr_db),
                        receiver_state,
                        cfo_hint=float(acquisition_cfo),
                    )
                frame = candidate_frame
            if frame is None:
                raise RuntimeError("未能生成 curriculum 训练帧。")
            selection = self._select_training_cir(
                phase=phase,
                frame=frame,
                acquisition_cir=acquisition_cir,
                max_delay=int(max_delay),
                snr_db=float(snr_db),
                sample_index=int(sample_index),
            )
            self.condition_cir_source_counts[selection.source] = self.condition_cir_source_counts.get(selection.source, 0) + 1
            cir = selection.cir
            snr_items.append(float(snr_db))
            rx_iq_items.append(torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1))
            bit_items.append(frame.bits.float())
            region_items.append(frame.model_region_ids.long())
            selected_tail = select_training_tail(
                frame_tail=frame.tail_symbols.to(torch.complex64),
                receiver_tail=receiver_state.soft_tail,
                mode=training_tail_mode,
            )
            if training_tail_mode == "corrupted_oracle":
                selected_tail = corrupt_training_tail(
                    selected_tail,
                    flip_probability=float(self.config.get("tail_flip_probability", 0.1)),
                    sample_index=int(sample_index),
                )
            tail_items.append(_pad_complex_1d(selected_tail, int(self.model.config.max_delay)))
            cir_items.append(
                _pad_complex_1d(
                    cir.to(device=self.device, dtype=torch.complex64),
                    int(self.model.config.max_delay) + 1,
                )
            )
            phase_feature_items.append(
                estimate_phase_residual_vector(
                    frame.receiver_view(),
                    cir,
                    selected_tail,
                    blocks=4,
                    cfo_hint=float(acquisition_cfo),
                )
            )
            adapt_symbol_items.append(frame.tx_symbols)
            adapt_masks.append(frame.adapt_mask)
            reward_masks.append(frame.reward_mask)
            data_masks.append(frame.data_mask)

        rx_iq = torch.stack(rx_iq_items).to(self.device).float()
        bits = torch.stack(bit_items).to(self.device)
        region_ids = torch.stack(region_items).to(self.device)
        soft_tail = torch.stack(tail_items).to(self.device)
        cir_batch = torch.stack(cir_items).to(self.device)
        adapt_symbols = torch.stack(adapt_symbol_items).to(self.device).to(torch.complex64)
        power = cir_batch.abs()
        condition = CIRCondition(
            complex_cir=cir_batch,
            support_probability=(power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)).to(torch.float32),
            noise_variance=torch.tensor([10.0 ** (-snr / 10.0) for snr in snr_items], device=self.device, dtype=torch.float32),
            confidence=torch.ones(batch_size, device=self.device),
            latent_residual=_phase_feature_latent(phase_feature_items, self.device),
        )
        logits = refine_logits_with_tail(
            self.model,
            rx_iq,
            condition,
            region_ids,
            soft_tail,
            modulation=None,
            passes=int(self.config.get("tail_refinement_passes", 0)),
            adapt_symbols=adapt_symbols,
            adapt_mask=torch.stack(adapt_masks).to(self.device),
        )
        adapt_mask = torch.stack(adapt_masks).to(self.device)
        reward_mask = torch.stack(reward_masks).to(self.device)
        data_mask = torch.stack(data_masks).to(self.device)
        return supervised_equalization_loss(
            logits,
            bits,
            adapt_mask,
            reward_mask,
            data_mask,
            data_margin_loss_weight=float(self.config.get("data_margin_loss_weight", 0.0)),
            data_margin=float(self.config.get("data_margin", 1.0)),
        )

    def _advance_training_receiver_state(
        self,
        frame,
        cir: torch.Tensor,
        snr_db: float,
        receiver_state: ReceiverState,
        cfo_hint: float | None = None,
    ) -> None:
        """无标签地模拟在线接收机对历史帧的 soft-tail 递推。"""

        tail = receiver_state.soft_tail.to(self.device).to(torch.complex64)
        rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).to(self.device).float()
        region_ids = frame.model_region_ids.unsqueeze(0).to(self.device).long()
        cir_batch = cir.reshape(1, -1)
        power = cir_batch.abs()
        phase_features = estimate_phase_residual_vector(
            frame.receiver_view(), cir, tail, blocks=4, cfo_hint=cfo_hint
        )
        condition = CIRCondition(
            complex_cir=cir_batch,
            support_probability=(power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)).to(torch.float32),
            noise_variance=torch.tensor([10.0 ** (-float(snr_db) / 10.0)], device=self.device, dtype=torch.float32),
            confidence=torch.ones(1, device=self.device),
            latent_residual=_phase_feature_latent([phase_features], self.device),
        )
        with torch.no_grad():
            from training.windowed_discrete_ppo import refine_logits_with_tail

            logits = refine_logits_with_tail(
                self.model,
                rx_iq,
                condition,
                region_ids,
                tail.unsqueeze(0),
                modulation=None,
                passes=int(self.config.get("tail_refinement_passes", 0)),
                adapt_symbols=frame.tx_symbols.to(self.device).unsqueeze(0).to(torch.complex64),
                adapt_mask=frame.adapt_mask.to(self.device).unsqueeze(0),
            )
        tail_len = receiver_state.soft_tail.numel()
        tail_logits = logits.squeeze(0)[-tail_len:]
        receiver_state.update_tail(torch.complex(torch.tanh(tail_logits / 2.0), torch.zeros_like(tail_logits)))

    def _select_training_cir(
        self,
        *,
        phase: CurriculumPhase,
        frame,
        acquisition_cir: torch.Tensor,
        max_delay: int,
        snr_db: float,
        sample_index: int,
    ) -> TrainingCIRSelection:
        """选择本样本训练时喂给神经均衡器的 CIR 条件。

        默认保持旧行为：早期 true-CIR 阶段使用真值，估计阶段使用 acquisition
        CIR。启用 `cir_condition_augmentation` 后，在配置的来源之间循环采样，
        让模型见到在线 DD-CIR 会产生的条件误差分布。
        """

        augmentation = self.config.get("cir_condition_augmentation", {})
        enabled = bool(augmentation.get("enabled", False))
        if not enabled:
            if phase.name in {"cir_level_a", "perfect_cir_level_a"}:
                return TrainingCIRSelection(_normalize_cir(frame.true_cir), "true")
            return TrainingCIRSelection(_normalize_cir(acquisition_cir), "acquisition")

        modes = [str(item) for item in augmentation.get("modes", ["true", "acquisition", "noisy", "dd_like"])]
        if not modes:
            modes = ["acquisition"]
        source = modes[int(sample_index) % len(modes)]
        if source == "true":
            return TrainingCIRSelection(_normalize_cir(frame.true_cir), source)
        if source == "acquisition":
            return TrainingCIRSelection(_normalize_cir(acquisition_cir), source)
        if source == "noisy":
            return TrainingCIRSelection(
                self._noisy_cir(acquisition_cir, float(augmentation.get("noise_std", 0.05)), int(sample_index)),
                source,
            )
        if source == "dd_like":
            return TrainingCIRSelection(
                self._dd_like_cir(
                    frame=frame,
                    acquisition_cir=acquisition_cir,
                    max_delay=int(max_delay),
                    sample_index=int(sample_index),
                    flip_probability=float(augmentation.get("dd_flip_probability", 0.02)),
                    blend_alpha=float(augmentation.get("dd_blend_alpha", 0.2)),
                ),
                source,
            )
        raise ValueError(f"未知 CIR condition augmentation 来源：{source}")

    def _noisy_cir(self, cir: torch.Tensor, noise_std: float, sample_index: int) -> torch.Tensor:
        """对 acquisition CIR 加复高斯扰动并重新归一化。"""

        vector = cir.reshape(-1).to(torch.complex64)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.get("curriculum_seed", 80_000)) + int(sample_index) * 1009 + 17)
        real = torch.randn(vector.shape, generator=generator, dtype=torch.float32).to(vector.device)
        imag = torch.randn(vector.shape, generator=generator, dtype=torch.float32).to(vector.device)
        noise = torch.complex(real, imag) * float(noise_std)
        return _normalize_cir(vector + noise)

    def _dd_like_cir(
        self,
        *,
        frame,
        acquisition_cir: torch.Tensor,
        max_delay: int,
        sample_index: int,
        flip_probability: float,
        blend_alpha: float,
    ) -> torch.Tensor:
        """构造离线 DD-like CIR 条件，模拟在线判决导向估计误差。

        Adapt Pilot 位置保持真导频；其余位置按固定随机种子翻转一小部分符号，
        再用整帧接收信号做 LS CIR 估计，最后与 acquisition CIR 混合。
        """

        target_device = acquisition_cir.device
        tx_estimate = frame.tx_symbols.detach().clone().to(target_device).to(torch.complex64)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(self.config.get("curriculum_seed", 80_000)) + int(sample_index) * 1619 + 31)
        random_values = torch.rand(tx_estimate.shape, generator=generator, dtype=torch.float32).to(tx_estimate.device)
        flip_mask = (random_values < float(flip_probability)) & (~frame.adapt_mask.to(target_device).to(torch.bool))
        tx_estimate[flip_mask] = -tx_estimate[flip_mask]
        dd_estimate = _estimate_cir_from_tx_symbols(
            frame.rx_symbols.to(target_device),
            tx_estimate,
            int(max_delay),
        )
        acquisition = _normalize_cir(acquisition_cir).to(target_device)
        blended = (1.0 - float(blend_alpha)) * acquisition + float(blend_alpha) * dd_estimate
        return _normalize_cir(blended)

    def _validate_level_b(self) -> dict[str, Any]:
        rows = []
        frames_per_config = int(self.config.get("validation_frames_per_config", 2))
        seeds = self.config.get("validation_seeds", [0, 1])
        for delay in self.config.get("main_delays", [20, 30, 40]):
            for snr_db in self.config.get("main_snrs", [0, 5, 10, 15]):
                bers = []
                for seed in seeds:
                    env = CommunicationEnvironment(
                        build_comm_env_config(
                            self.config,
                            level="B",
                            snr_db=float(snr_db),
                            seed=10_000 + int(seed),
                            max_delay=int(delay),
                            total_pilot=int(self.config.get("pilot_total", 128)),
                            pilot_layout=str(self.config.get("pilot_layout", "prefix")),
                            impairment_profile=str(self.config.get("impairment_profile", "clean")),
                            state_split=self.config.get("offline_state_split"),
                        )
                    )
                    start = env.reset_episode()
                    receiver_state = ReceiverState(start.initial_soft_tail)
                    for _ in range(frames_per_config):
                        frame = env.next_frame()
                        result = perfect_csi_bpsk_refine_detect(
                            frame.rx_symbols,
                            frame.true_cir,
                            receiver_state.soft_tail,
                            torch.tensor(10.0 ** (-float(snr_db) / 10.0)),
                            cg_iterations=int(self.config.get("perfect_cir_cg_iterations", 64)),
                            refine_iterations=int(self.config.get("perfect_cir_refine_iterations", 2)),
                        )
                        receiver_state.update_tail(result.soft_tail)
                        bers.append(bit_error_rate(result.logits[frame.data_mask], frame.bits[frame.data_mask]))
                rows.append(
                    {
                        "level": "B",
                        "delay": int(delay),
                        "snr_db": float(snr_db),
                        "frames": frames_per_config,
                        "seeds": list(seeds),
                        "ber_data": float(sum(bers) / max(1, len(bers))),
                        "max_frame_ber_data": float(max(bers)) if bers else 1.0,
                    }
                )
        threshold = float(self.config.get("perfect_cir_gate_ber", 0.01))
        gate_pass = all(row["ber_data"] < threshold for row in rows)
        return {
            "selection_metric": "mean_level_b_ber_data",
            "gate": "perfect_cir_bpsk_refine",
            "gate_threshold": threshold,
            "gate_pass": gate_pass,
            "mean_ber_data": float(sum(row["ber_data"] for row in rows) / max(1, len(rows))),
            "per_config": rows,
        }

    def _validate_offline_nn(self) -> dict[str, Any]:
        """验证当前神经均衡器本身的离线 BER。

        这是训练是否有效的核心指标；强诊断检测器通过不代表神经模型已学会。
        """

        from training.meta_training import estimate_acquisition_cir_for_profile

        was_training = self.model.training
        self.model.eval()
        frames_per_config = int(self.config.get("model_validation_frames_per_config", 1))
        seeds = self.config.get("model_validation_seeds", [0])
        if self.config.get("channel_profile") in {"eme_measurement_v1", "eme_long_memory_v2"}:
            configs = self.config.get(
                "model_validation_configs",
                [
                    {
                        "level": "B",
                        "delay": int(delay),
                        "snr_db": float(snr_db),
                        "pilot_total": int(self.config.get("pilot_total", 128)),
                        "pilot_layout": self.config.get("pilot_layout", "prefix"),
                        "cir": "estimated",
                    }
                    for delay in self.config.get("main_delays", [self.config["max_delay"]])
                    for snr_db in self.config.get("main_snrs", [0, 5, 10, 15])
                ],
            )
        else:
            configs = self.config.get(
                "model_validation_configs",
                [
                    {"level": "A", "delay": 20, "snr_db": 15.0, "pilot_total": 160, "pilot_layout": self.config.get("pilot_layout", "prefix"), "cir": "true"},
                    {"level": "B", "delay": 20, "snr_db": 10.0, "pilot_total": int(self.config.get("pilot_total", 128)), "pilot_layout": self.config.get("pilot_layout", "prefix"), "cir": "estimated"},
                ],
            )
        rows = []
        seed_base = int(self.config.get("model_validation_seed_base", 90_000))
        with torch.no_grad():
            for item in configs:
                bers = []
                reward_bers = []
                adapt_bers = []
                sequence_state = bool(item.get("sequence_state", self.config.get("model_validation_sequence_state", False)))
                cir_mode = str(item.get("cir", "estimated"))
                tail_mode = str(item.get("tail_mode", "soft" if sequence_state else "oracle"))
                for seed in seeds:
                    env = CommunicationEnvironment(
                        build_comm_env_config(
                            self.config,
                            level=str(item["level"]),
                            snr_db=float(item["snr_db"]),
                            seed=seed_base + int(seed),
                            max_delay=int(item["delay"]),
                            total_pilot=int(item["pilot_total"]),
                            pilot_layout=str(item["pilot_layout"]),
                            impairment_profile=str(item.get("impairment_profile", self.config.get("impairment_profile", "clean"))),
                            state_split=item.get("state_split", self.config.get("offline_state_split")),
                        )
                    )
                    start = env.reset_episode()
                    validation_impairment = str(
                        item.get("impairment_profile", self.config.get("impairment_profile", "clean"))
                    )
                    if validation_impairment == "clean":
                        acquisition_cir = estimate_acquisition_cir_for_profile(
                            start.acquisition,
                            int(item["delay"]),
                            "clean",
                        )
                        acquisition_cfo = 0.0
                    else:
                        from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo

                        acquisition_cir, acquisition_cfo = estimate_acquisition_cir_with_cfo(
                            start.acquisition,
                            int(item["delay"]),
                            cfo_limit=float(self.config.get("acquisition_cfo_limit", 0.004)),
                        )
                    receiver_state = ReceiverState(start.initial_soft_tail.to(self.device).to(torch.complex64))
                    cir_state = acquisition_cir.to(self.device).to(torch.complex64)
                    for _ in range(frames_per_config):
                        frame = env.next_frame()
                        cir = self._select_validation_cir(
                            cir_mode=cir_mode,
                            cir_state=cir_state,
                            frame=frame,
                        )
                        rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).to(self.device).float()
                        region_ids = frame.model_region_ids.unsqueeze(0).to(self.device).long()
                        if sequence_state:
                            soft_tail = receiver_state.soft_tail.unsqueeze(0).to(self.device).to(torch.complex64)
                        else:
                            soft_tail = frame.tail_symbols.unsqueeze(0).to(self.device).to(torch.complex64)
                        cir_b = cir.unsqueeze(0).to(self.device).to(torch.complex64)
                        power = cir_b.abs()
                        phase_tail = select_validation_tail(
                            frame_tail=frame.tail_symbols,
                            receiver_tail=receiver_state.soft_tail,
                            sequence_state=sequence_state,
                        )
                        phase_features = estimate_phase_residual_vector(
                            frame.receiver_view(),
                            cir,
                            phase_tail,
                            blocks=4,
                            cfo_hint=float(acquisition_cfo),
                        )
                        condition = CIRCondition(
                            complex_cir=cir_b,
                            support_probability=(power / power.sum(dim=1, keepdim=True).clamp_min(1e-8)).to(torch.float32),
                            noise_variance=torch.full((1,), 10.0 ** (-float(item["snr_db"]) / 10.0), device=self.device),
                            confidence=torch.ones(1, device=self.device),
                            latent_residual=_phase_feature_latent([phase_features], self.device),
                        )
                        from training.windowed_discrete_ppo import refine_logits_with_tail

                        logits = refine_logits_with_tail(
                            self.model,
                            rx_iq,
                            condition,
                            region_ids,
                            soft_tail,
                            modulation=None,
                            passes=int(self.config.get("tail_refinement_passes", 0)),
                            adapt_symbols=frame.tx_symbols.to(self.device).unsqueeze(0).to(torch.complex64),
                            adapt_mask=frame.adapt_mask.to(self.device).unsqueeze(0),
                        )
                        data_mask = frame.data_mask.to(self.device)
                        reward_mask = frame.reward_mask.to(self.device)
                        adapt_mask = frame.adapt_mask.to(self.device)
                        bits = frame.bits.to(self.device)
                        frame_logits = logits.squeeze(0)
                        bers.append(bit_error_rate(frame_logits[data_mask], bits[data_mask]))
                        reward_bers.append(bit_error_rate(frame_logits[reward_mask], bits[reward_mask]))
                        adapt_bers.append(bit_error_rate(frame_logits[adapt_mask], bits[adapt_mask]))
                        if sequence_state:
                            tail_len = receiver_state.soft_tail.numel()
                            tail_logits = frame_logits[-tail_len:].detach()
                            receiver_state.update_tail(torch.complex(torch.tanh(tail_logits / 2.0), torch.zeros_like(tail_logits)))
                            if cir_mode in {"decision_directed", "dd", "dd_cir"}:
                                cir_state = decision_directed_cir_update(
                                    _frame_to_device_for_validation(frame, self.device),
                                    frame_logits.detach(),
                                    int(item["delay"]),
                                    cir_state,
                                    alpha=float(item.get("cir_alpha", self.config.get("model_validation_cir_alpha", 0.2))),
                                ).to(self.device)
                rows.append(
                    {
                        "level": str(item["level"]),
                        "delay": int(item["delay"]),
                        "snr_db": float(item["snr_db"]),
                        "pilot_total": int(item["pilot_total"]),
                        "pilot_layout": str(item["pilot_layout"]),
                        "cir": cir_mode,
                        "cir_update_uses_data_labels": False,
                        "tail_mode": tail_mode,
                        "sequence_state": sequence_state,
                        "seed_base": int(seed_base),
                        "frames": int(frames_per_config),
                        "seeds": list(seeds),
                        "ber_data": float(sum(bers) / max(1, len(bers))),
                        "ber_reward_pilot": float(sum(reward_bers) / max(1, len(reward_bers))),
                        "ber_adapt_pilot": float(sum(adapt_bers) / max(1, len(adapt_bers))),
                    }
                )
        self.model.train(was_training)
        threshold = float(self.config.get("offline_nn_gate_ber", 0.2))
        return {
            "metric": "offline_nn_ber_data",
            "gate_threshold": threshold,
            "gate_pass": all(row["ber_data"] < threshold for row in rows),
            "mean_ber_data": float(sum(row["ber_data"] for row in rows) / max(1, len(rows))),
            "rows": rows,
        }

    def _select_validation_cir(self, *, cir_mode: str, cir_state: torch.Tensor, frame) -> torch.Tensor:
        """选择离线验证帧使用的 CIR 条件。"""

        if cir_mode in {"true", "oracle"}:
            return frame.true_cir
        return cir_state


def _frame_to_device_for_validation(frame, device: torch.device):
    """把验证帧迁移到设备，供 DD-CIR 更新使用。"""

    from dataclasses import replace

    return replace(
        frame,
        bits=frame.bits.to(device),
        tx_symbols=frame.tx_symbols.to(device),
        rx_symbols=frame.rx_symbols.to(device),
        adapt_mask=frame.adapt_mask.to(device),
        reward_mask=frame.reward_mask.to(device),
        data_mask=frame.data_mask.to(device),
        model_region_ids=frame.model_region_ids.to(device),
        tail_symbols=frame.tail_symbols.to(device) if frame.tail_symbols is not None else None,
        true_cir=frame.true_cir.to(device) if frame.true_cir is not None else None,
    )
