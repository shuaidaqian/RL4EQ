# -*- coding: utf-8 -*-
"""物理引导展开式整帧神经均衡器。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from agent.cir_estimator import CIRCondition
from agent.modulation import ModulationState
from agent.peft import PEFTRegistry, mark_peft_group
from env.linear_operator import LinearChannelOperator


@dataclass(frozen=True)
class UnfoldedConfig:
    frame_len: int = 512
    max_delay: int = 40
    iterations: int = 4
    d_model: int = 96
    num_heads: int = 4
    adapter_rank: int = 8
    lora_rank: int = 8
    conditioner_uses_cir_summary: bool = False
    enable_phase_correction_branch: bool = False
    phase_correction_segments: int = 4
    phase_correction_initial_scale: float = 0.0
    analytic_logit_skip_scale: float = 0.0
    physics_warm_start_iterations: int = 0
    physics_warm_start_scale: float = 1.0
    neural_residual_scale: float = 1.0
    pilot_conditioned: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "UnfoldedConfig":
        return cls(**data)


class BottleneckAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank)
        self.up = nn.Linear(rank, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(torch.relu(self.down(x)))


class LoRAResidual(nn.Module):
    def __init__(self, d_model: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class DenoiserBlock(nn.Module):
    def __init__(self, config: UnfoldedConfig):
        super().__init__()
        self.attn = nn.MultiheadAttention(config.d_model, config.num_heads, batch_first=True)
        self.attn_lora = LoRAResidual(config.d_model, config.lora_rank)
        self.norm1 = nn.LayerNorm(config.d_model)
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Linear(config.d_model * 2, config.d_model),
        )
        self.ffn_lora = LoRAResidual(config.d_model, config.lora_rank)
        self.adapter = BottleneckAdapter(config.d_model, config.adapter_rank)
        self.norm2 = nn.LayerNorm(config.d_model)
        mark_peft_group(self.attn_lora, "attention_lora")
        mark_peft_group(self.ffn_lora, "ffn_lora")
        mark_peft_group(self.adapter, "adapter")

    def forward(
        self,
        x: torch.Tensor,
        adapter_gate: torch.Tensor | None = None,
        lora_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_lora = self.attn_lora(x)
        ffn_lora = self.ffn_lora(x)
        adapter = self.adapter(x)
        if lora_scale is not None:
            attn_lora = attn_lora * lora_scale
            ffn_lora = ffn_lora * lora_scale
        if adapter_gate is not None:
            adapter = adapter * adapter_gate
        x = self.norm1(x + attn_out + attn_lora)
        x = self.norm2(x + self.ffn(x) + ffn_lora + adapter)
        return x


class UnfoldedEqualizer(nn.Module):
    """显式 H/H^H 迭代与非因果 Transformer denoiser 结合的块均衡器。"""

    def __init__(self, config: UnfoldedConfig | None = None):
        super().__init__()
        self.config = config or UnfoldedConfig()
        self.feature_proj = nn.Linear(4, self.config.d_model)
        self.region_embedding = nn.Embedding(2, self.config.d_model)
        self.pilot_encoder = None
        if self.config.pilot_conditioned:
            self.pilot_encoder = nn.Sequential(
                nn.Conv1d(5, self.config.d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(self.config.d_model, self.config.d_model, kernel_size=1),
            )
        self.phase_correction = None
        if self.config.enable_phase_correction_branch:
            self.phase_correction = nn.Sequential(
                nn.Linear(96, self.config.d_model),
                nn.Tanh(),
                nn.Linear(self.config.d_model, int(self.config.phase_correction_segments)),
            )
            nn.init.zeros_(self.phase_correction[-1].weight)
            nn.init.zeros_(self.phase_correction[-1].bias)
            self.phase_correction_scale = nn.Parameter(
                torch.tensor(float(self.config.phase_correction_initial_scale), dtype=torch.float32)
            )
            mark_peft_group(self.phase_correction, "phase")
            setattr(self.phase_correction_scale, "_peft_group", "phase")
        else:
            self.register_parameter("phase_correction_scale", None)
        conditioner_input_dim = 96
        if self.config.pilot_conditioned:
            conditioner_input_dim += self.config.d_model
        if self.config.conditioner_uses_cir_summary:
            conditioner_input_dim += 2 * (self.config.max_delay + 1) + 2
        self.conditioner = nn.Sequential(
            nn.Linear(conditioner_input_dim, self.config.d_model * 2),
            nn.Tanh(),
            nn.Linear(self.config.d_model * 2, self.config.d_model * 2),
        )
        self.blocks = nn.ModuleList([DenoiserBlock(self.config) for _ in range(3)])
        self.head = nn.Linear(self.config.d_model, 1)
        if int(self.config.physics_warm_start_iterations) > 0:
            # 物理检测器先提供可用判决，神经 head 只学习物理模型失配残差。
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
        self.alpha = nn.Parameter(torch.full((self.config.iterations,), 0.2))
        self.damping = nn.Parameter(torch.full((self.config.iterations,), 0.2))
        mark_peft_group(self.conditioner, "conditioner_film")
        mark_peft_group(self.head, "head")
        self.peft = PEFTRegistry(self)
        self.set_trainable_groups(set())

    def forward(
        self,
        rx_iq: torch.Tensor,
        condition: CIRCondition,
        region_ids: torch.Tensor,
        soft_tail: torch.Tensor,
        modulation: ModulationState | None = None,
        adapt_symbols: torch.Tensor | None = None,
        adapt_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rx_iq = self._apply_phase_correction(rx_iq, condition)
        pilot_context = None
        if self.config.pilot_conditioned:
            pilot_context = self._pilot_context(rx_iq, adapt_symbols, adapt_mask)
        rx_complex = torch.complex(rx_iq[..., 0], rx_iq[..., 1]).to(torch.complex64)
        physics_logits = self._physics_warm_start_logits(rx_complex, condition, soft_tail)
        operator = LinearChannelOperator(frame_len=rx_complex.shape[1], max_delay=condition.complex_cir.shape[1] - 1)
        soft_symbols = torch.zeros_like(rx_complex)
        logits = torch.zeros_like(rx_complex.real)
        for layer in range(self.config.iterations):
            residual = rx_complex - operator.forward(soft_symbols, condition.complex_cir, soft_tail)
            gradient = operator.adjoint(residual, condition.complex_cir)
            proposal = soft_symbols + self.alpha[layer].to(gradient.dtype) * gradient
            features = torch.stack((proposal.real, proposal.imag, residual.real, residual.imag), dim=-1)
            hidden = self.feature_proj(features) + self.region_embedding(region_ids.clamp_min(0).clamp_max(1))
            hidden = self._apply_film(hidden, condition, modulation, pilot_context)
            for block_index, block in enumerate(self.blocks):
                adapter_gate = None
                lora_scale = None
                if modulation is not None:
                    adapter_gate = modulation.adapter_gates[
                        min(block_index, modulation.adapter_gates.numel() - 1)
                    ].to(hidden.device, hidden.dtype)
                    lora_scale = modulation.lora_scales[
                        min(block_index, modulation.lora_scales.numel() - 1)
                    ].to(hidden.device, hidden.dtype)
                hidden = block(hidden, adapter_gate=adapter_gate, lora_scale=lora_scale)
            neural_logits = self.head(hidden).squeeze(-1)
            logits = self._apply_head_modulation(
                float(self.config.neural_residual_scale) * neural_logits
                + self._analytic_logit_skip(proposal, condition)
                + physics_logits,
                modulation,
            )
            damping = torch.sigmoid(self.damping[layer])
            soft_update = torch.complex(torch.tanh(logits), torch.zeros_like(logits))
            soft_symbols = (1.0 - damping) * soft_update + damping * soft_symbols
        return logits, torch.sigmoid(logits)

    def _physics_warm_start_logits(
        self,
        rx_complex: torch.Tensor,
        condition: CIRCondition,
        soft_tail: torch.Tensor,
    ) -> torch.Tensor:
        """用接收端可见 CIR 和 soft tail 计算可微的长记忆物理 warm-start。"""

        iterations = int(self.config.physics_warm_start_iterations)
        if iterations <= 0:
            return torch.zeros_like(rx_complex.real)
        cir = condition.complex_cir.to(device=rx_complex.device, dtype=torch.complex64)
        tail = soft_tail.to(device=rx_complex.device, dtype=torch.complex64)
        if tail.ndim == 1:
            tail = tail.unsqueeze(0)
        operator = LinearChannelOperator(
            frame_len=rx_complex.shape[1],
            max_delay=cir.shape[1] - 1,
        )
        zero = torch.zeros_like(rx_complex)
        tail_contribution = operator.forward(zero, cir, tail)
        rhs = operator.adjoint(rx_complex - tail_contribution, cir)
        noise = condition.noise_variance.to(
            device=rx_complex.device,
            dtype=rx_complex.real.dtype,
        ).reshape(-1, 1).clamp_min(1e-6)

        def normal_matvec(vector: torch.Tensor) -> torch.Tensor:
            return operator.adjoint(
                operator.forward(vector, cir, torch.zeros_like(tail)),
                cir,
            ) + noise.to(vector.dtype) * vector

        estimate = torch.zeros_like(rhs)
        residual = rhs - normal_matvec(estimate)
        direction = residual.clone()
        residual_energy = torch.sum(torch.conj(residual) * residual, dim=1, keepdim=True).real
        for _ in range(iterations):
            mat_direction = normal_matvec(direction)
            denominator = torch.sum(
                torch.conj(direction) * mat_direction,
                dim=1,
                keepdim=True,
            ).real.clamp_min(1e-8)
            step = residual_energy / denominator
            estimate = estimate + step.to(estimate.dtype) * direction
            residual = residual - step.to(residual.dtype) * mat_direction
            next_energy = torch.sum(torch.conj(residual) * residual, dim=1, keepdim=True).real
            beta = next_energy / residual_energy.clamp_min(1e-8)
            direction = residual + beta.to(direction.dtype) * direction
            residual_energy = next_energy
        logits = 2.0 * estimate.real / noise
        scale = float(self.config.physics_warm_start_scale)
        return (float(scale) * logits).clamp(-40.0, 40.0)

    def _analytic_logit_skip(self, proposal: torch.Tensor, condition: CIRCondition) -> torch.Tensor:
        """给神经 head 叠加 BPSK 解析 LLR warm-start，默认关闭。"""

        scale = float(self.config.analytic_logit_skip_scale)
        if scale == 0.0:
            return torch.zeros_like(proposal.real)
        noise = condition.noise_variance.reshape(-1, 1).to(device=proposal.device, dtype=proposal.real.dtype)
        return (scale * 2.0 * proposal.real / noise.clamp_min(1e-4)).clamp(-40.0, 40.0)

    def _apply_phase_correction(self, rx_iq: torch.Tensor, condition: CIRCondition) -> torch.Tensor:
        """根据 phase-vector conditioner 对整帧接收 I/Q 做显式相位校正。"""

        if not self.config.enable_phase_correction_branch:
            return rx_iq
        latent = condition.latent_residual
        if latent.shape[1] < 96:
            latent = torch.nn.functional.pad(latent, (0, 96 - latent.shape[1]))
        latent = latent[:, :96].to(rx_iq.device, rx_iq.dtype)
        batch, frame_len, _ = rx_iq.shape
        segment_count = max(1, int(self.config.phase_correction_segments))
        phase0_values = []
        cfo_values = []
        for segment in range(segment_count):
            offset = segment * 4
            if offset + 1 < latent.shape[1]:
                phase0_values.append(latent[:, offset])
                cfo_values.append(latent[:, offset + 1])
            else:
                phase0_values.append(torch.zeros(batch, device=rx_iq.device, dtype=rx_iq.dtype))
                cfo_values.append(torch.zeros(batch, device=rx_iq.device, dtype=rx_iq.dtype))
        phase0 = torch.stack(phase0_values, dim=1)
        cfo = torch.stack(cfo_values, dim=1)
        if self.phase_correction is not None:
            phase0 = phase0 + self.phase_correction(latent)
        # CFO 与公共相位在当前 profile 中是慢状态；按 block 中位数聚合，
        # 避免某个低信噪比 Pilot block 把整帧校正带偏。
        phase0_global = torch.median(phase0, dim=1).values
        cfo_global = torch.median(cfo, dim=1).values
        positions = torch.arange(frame_len, device=rx_iq.device, dtype=rx_iq.dtype)
        phase0_t = phase0_global.unsqueeze(1)
        cfo_t = cfo_global.unsqueeze(1)
        phase = phase0_t + 2.0 * torch.pi * cfo_t * positions.unsqueeze(0)
        scale = self.phase_correction_scale.to(rx_iq.device, rx_iq.dtype)
        phase = phase * scale
        rx_complex = torch.complex(rx_iq[..., 0], rx_iq[..., 1]).to(torch.complex64)
        corrected = rx_complex * torch.exp(-1j * phase.to(torch.float32)).to(torch.complex64)
        return torch.stack((corrected.real, corrected.imag), dim=-1).to(rx_iq.dtype)

    def _apply_film(
        self,
        hidden: torch.Tensor,
        condition: CIRCondition,
        modulation: ModulationState | None = None,
        pilot_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        latent = condition.latent_residual
        if latent.shape[1] < 96:
            latent = torch.nn.functional.pad(latent, (0, 96 - latent.shape[1]))
        latent = latent[:, :96].to(hidden.dtype)
        conditioner_features = latent
        if self.config.conditioner_uses_cir_summary:
            conditioner_features = torch.cat(
                [
                    latent,
                    self._pad_condition_vector(torch.abs(condition.complex_cir), self.config.max_delay + 1, hidden),
                    self._pad_condition_vector(condition.support_probability, self.config.max_delay + 1, hidden),
                    torch.log1p(condition.noise_variance).reshape(-1, 1).to(hidden.device, hidden.dtype),
                    condition.confidence.reshape(-1, 1).to(hidden.device, hidden.dtype),
                ],
                dim=-1,
            )
        if self.config.pilot_conditioned:
            if pilot_context is None:
                pilot_context = torch.zeros(
                    hidden.shape[0], self.config.d_model,
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
            conditioner_features = torch.cat(
                [conditioner_features, pilot_context.to(hidden.device, hidden.dtype)],
                dim=-1,
            )
        gamma_beta = self.conditioner(conditioner_features)
        gamma, beta = gamma_beta.chunk(2, dim=-1)
        if modulation is not None:
            scale = modulation.film_residual_scale.to(hidden.device, hidden.dtype)
            gamma = gamma * (1.0 + scale)
            beta = beta * (1.0 + scale)
        return hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def _pilot_context(
        self,
        rx_iq: torch.Tensor,
        adapt_symbols: torch.Tensor | None,
        adapt_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """从已知 Adapt Pilot 提取不读取隐藏标签的帧级条件。"""

        if self.pilot_encoder is None:
            raise RuntimeError("pilot_conditioned 未启用。")
        if adapt_symbols is None or adapt_mask is None:
            return torch.zeros(
                rx_iq.shape[0], self.config.d_model,
                device=rx_iq.device,
                dtype=rx_iq.dtype,
            )
        mask = adapt_mask.to(device=rx_iq.device, dtype=torch.bool)
        if mask.ndim == 1:
            mask = mask.unsqueeze(0)
        symbols = adapt_symbols.to(device=rx_iq.device)
        if symbols.ndim == 1:
            symbols = symbols.unsqueeze(0)
        mask_float = mask.to(rx_iq.dtype)
        symbols_iq = torch.stack((symbols.real, symbols.imag), dim=-1).to(rx_iq.dtype)
        features = torch.cat(
            [rx_iq * mask_float.unsqueeze(-1), symbols_iq * mask_float.unsqueeze(-1), mask_float.unsqueeze(-1)],
            dim=-1,
        )
        encoded = self.pilot_encoder(features.transpose(1, 2)).transpose(1, 2)
        weights = mask_float.unsqueeze(-1)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def _pad_condition_vector(self, value: torch.Tensor, target_len: int, hidden: torch.Tensor) -> torch.Tensor:
        """把帧级条件向量补齐到模型配置的 max_delay 长度。"""

        vector = value.to(hidden.device, hidden.dtype)
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        if vector.shape[1] < int(target_len):
            vector = torch.nn.functional.pad(vector, (0, int(target_len) - vector.shape[1]))
        return vector[:, : int(target_len)]

    def _apply_head_modulation(
        self,
        logits: torch.Tensor,
        modulation: ModulationState | None = None,
    ) -> torch.Tensor:
        if modulation is None:
            return logits
        temperature = modulation.head_temperature.to(logits.device, logits.dtype).clamp(0.5, 2.0)
        bias = modulation.head_bias.to(logits.device, logits.dtype).clamp(-1.0, 1.0)
        return logits * temperature + bias

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def set_trainable_groups(self, groups: set[str]) -> None:
        resolved = self.peft.resolve(groups)
        for parameter in self.parameters():
            parameter.requires_grad_(getattr(parameter, "_peft_group", None) in resolved)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.trainable_parameters())
