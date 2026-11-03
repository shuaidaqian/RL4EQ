# -*- coding: utf-8 -*-
"""同波形可适配性开发审计；可部署输入与真状态诊断在接口处隔离。"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from agent.cir_estimator import condition_from_cir
from agent.neural_channel_state import known_pilot_equations
from agent.unfolded_equalizer import UnfoldedEqualizer
from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config, validate_model_dimensions
from evaluation.joint_pilot_state import PilotBlock, joint_posterior
from evaluation.pilot_transfer import _load_branch, paired_interval
from training.deterministic_adaptation import DeterministicMemoryAdapter, DeterministicSettings
from training.pilot_transfer import TransferContext, context_logits, frame_to_device, make_context


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha256(value):
    if value is None:
        return None
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def deployable_frame(frame):
    """兼容旧 Frame，但只保留已知 Pilot 与接收波形；在控制开始前剥离标签。"""
    view = frame.receiver_view()
    return replace(frame, bits=torch.zeros_like(frame.bits),
        tx_symbols=view.adapt_symbols + view.reward_symbols, true_cir=None, tail_symbols=None)


@dataclass(frozen=True)
class DiagnosticTruth:
    """不含发送 Data、bit 或真实 tail 的隔离诊断状态。"""

    cir: torch.Tensor
    phase: torch.Tensor
    lookahead_phase: torch.Tensor | None = None
    lookahead_cir: torch.Tensor | None = None


class AuditEnvironment(CommunicationEnvironment):
    """按生成时刻缓存诊断快照，避免 lookahead 使当前真状态错位。"""

    def __init__(self, config):
        super().__init__(config)
        self.diagnostic_truth = {}

    def reset_episode(self):
        self.diagnostic_truth.clear()
        return super().reset_episode()

    def _transmit_next_frame(self):
        frame = super()._transmit_next_frame()
        self.diagnostic_truth[frame.frame_index] = DiagnosticTruth(
            self.channel.last_cir_used(), self.channel.last_phase_used())
        return frame

    def truth_for(self, frame):
        current = self.diagnostic_truth[frame.frame_index]
        if frame.lookahead_rx_symbols is None:
            return current
        future = self.diagnostic_truth[frame.lookahead_frame_index]
        count = frame.lookahead_rx_symbols.numel()
        return replace(current, lookahead_phase=future.phase[:count], lookahead_cir=future.cir)


class ExactStateDiagnostic(UnfoldedEqualizer):
    """同权重接收机，仅隔离替换同步/CIR状态，不向可部署网络传真值。"""

    @classmethod
    def from_model(cls, model):
        result = cls(copy.deepcopy(model.config))
        if model.online_phase_trend_adapter is not None:
            result.attach_online_phase_trend_adapter()
        result.to(next(model.parameters()).device)
        result.load_state_dict(model.state_dict(), strict=True)
        result.eval()
        for value in result.parameters():
            value.requires_grad_(False)
        result._diagnostic_truth = None
        return result

    def _apply_phase_correction(self, rx_iq, condition):
        truth = self._diagnostic_truth
        if truth is None:
            raise RuntimeError("真状态诊断缺少隔离快照。")
        phase = truth.phase
        if truth.lookahead_phase is not None:
            phase = torch.cat((phase, truth.lookahead_phase))
        phase = phase.to(rx_iq.device)
        if phase.numel() != rx_iq.shape[1]:
            raise ValueError("诊断相位与实际接收窗口不匹配。")
        corrected = torch.complex(rx_iq[..., 0], rx_iq[..., 1]) * torch.exp(-1j * phase)
        return torch.stack((corrected.real, corrected.imag), -1).to(rx_iq.dtype)

    def _channel_state_condition(self, corrected_iq, condition, adapt_symbols, adapt_mask, frame_index):
        return condition

    @torch.no_grad()
    def diagnostic_logits(self, context, truth):
        self._diagnostic_truth = truth
        try:
            condition = condition_from_cir(truth.cir.to(context.tail.device),
                -10 * math.log10(float(context.condition.noise_variance.flatten()[0])))
            return context_logits(self, replace(context, condition=condition), visible="full_rx").detach()
        finally:
            self._diagnostic_truth = None


@torch.no_grad()
def corrected_block(model, context):
    view = context.frame.receiver_view()
    rx = torch.where(view.adapt_mask, view.rx_symbols, torch.zeros_like(view.rx_symbols))
    iq = torch.stack((rx.real, rx.imag), -1).unsqueeze(0).float()
    corrected = model._apply_phase_correction(iq, context.condition)
    design, values = known_pilot_equations(torch.complex(corrected[0, :, 0], corrected[0, :, 1]),
        view.adapt_symbols, view.adapt_mask, view.adapt_mask, context.tail.numel())
    return PilotBlock(design, values, float(context.condition.noise_variance.flatten()[0]), context.frame.frame_index)


class LegalStateTracker:
    """合法 Pilot 的全协方差递推；每条新帧似然只消费一次。"""

    def __init__(self, cir, noise, acquisition_rows, rho):
        if acquisition_rows < 1 or not 0 < rho <= 1:
            raise ValueError("状态递推需要合法 acquisition 行数和相关系数。")
        self.mean = cir.detach().to(torch.complex128)
        variance = float(noise) / acquisition_rows
        self.covariance = torch.eye(cir.numel(), device=cir.device, dtype=torch.complex128) * variance
        # 只从 acquisition 幅度和声明模型设定过程噪声，绝不使用真实 PDP。
        self.stationary_power = cir.abs().square().to(torch.float64) + variance
        self.rho, self.last_index = float(rho), -1

    @torch.no_grad()
    def context(self, model, frame, cfo, tail, snr):
        if frame.frame_index != self.last_index + 1:
            raise ValueError("状态递推必须从会话开始逐帧消费，禁止重复或跳帧。")
        transition = self.rho
        predicted = transition * self.mean
        covariance = transition ** 2 * self.covariance + torch.diag(
            ((1 - transition ** 2) * self.stationary_power).to(torch.complex128))
        pre = make_context(frame, predicted.to(torch.complex64), cfo, tail, snr)
        block = corrected_block(model, pre)
        self.mean, self.covariance = joint_posterior(predicted, covariance, [block])
        self.last_index = frame.frame_index
        condition = condition_from_cir(self.mean.to(torch.complex64), snr,
            phase_features=pre.condition.latent_residual)
        return replace(pre, condition=condition), block


def information_metrics(design):
    """只描述局部信息几何；欠定时完整参数空间条件数记为空。"""
    singular = torch.linalg.svdvals(design.detach().to(device="cpu", dtype=torch.complex128))
    tolerance = max(design.shape) * torch.finfo(torch.float32).eps * singular.max()
    rank = int((singular > tolerance).sum())
    positive = singular[singular > tolerance]
    return dict(rows=design.shape[0], columns=design.shape[1], rank=rank,
        rank_relative_tolerance=max(design.shape) * torch.finfo(torch.float32).eps,
        singular_values=[float(value) for value in singular],
        singular_max=float(singular.max()), singular_min_nonzero=float(positive.min()) if rank else None,
        identifiable_condition_number=float(positive.max() / positive.min()) if rank else None,
        full_column_condition_number=float(singular.max() / singular.min()) if rank == design.shape[1] else None)


def memory_jacobian_metrics(model, context, design):
    """对当前独立 tap 记忆精确计算局部 Pilot Jacobian；不作全局可辨识推断。"""
    parameter = model.channel_state.memory.weight
    previous = parameter.requires_grad
    parameter.requires_grad_(True)
    try:
        with torch.enable_grad():
            view = context.frame.receiver_view()
            iq = torch.stack((view.rx_symbols.real, view.rx_symbols.imag), -1).unsqueeze(0).float()
            corrected = model._apply_phase_correction(iq, context.condition)
            state = model._channel_state_condition(corrected, context.condition,
                view.adapt_symbols.unsqueeze(0), view.adapt_mask.unsqueeze(0), context.frame.frame_index).complex_cir[0]
            dr = torch.autograd.grad(state.real.sum(), parameter, retain_graph=True)[0][:, 0]
            di = torch.autograd.grad(state.imag.sum(), parameter)[0][:, 1]
        real, imag = design.real, design.imag
        jacobian = torch.cat((torch.cat((real * dr, -imag * di), 1),
                              torch.cat((imag * dr, real * di), 1)), 0)
        return information_metrics(jacobian)
    finally:
        parameter.requires_grad_(previous)


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def _ber(logits, frame):
    return float(((logits[frame.data_mask] >= 0) != frame.bits[frame.data_mask]).float().mean())


@torch.no_grad()
def fixed_candidate_probe(models, context, physical, original, *, seed, snr, accepted):
    """首帧一次候选只读评分，未来指标不返回持续更新控制器。"""
    return dict(seed=seed, snr_db=snr, source_frame=0, target_frame=original.frame_index,
        offset=original.frame_index, accepted_at_source=accepted, ber_P=_ber(physical, original),
        **{"ber_"+name: _ber(context_logits(model, context, visible="full_rx"), original)
           for name, model in models.items()},
        rx_sha256=tensor_sha256(context.frame.rx_symbols),
        lookahead_rx_sha256=tensor_sha256(context.frame.lookahead_rx_symbols),
        diagnostic_only=True, future_metrics_used_for_control=False)


def summarize(rows, seeds, snrs):
    result = []
    for snr in snrs:
        active = [row for row in rows if row["snr_db"] == snr]
        arms = {}
        for arm in ("F", "P", "N", "T"):
            metric = "ber_" + arm
            gains = [float(np.mean([row["ber_P"] - row[metric] for row in active if row["seed"] == seed])) for seed in seeds]
            arms[arm] = dict(ber_data=float(np.mean([row[metric] for row in active])),
                gain_vs_P=paired_interval(gains),
                per_seed={str(seed): float(np.mean([row[metric] for row in active if row["seed"] == seed])) for seed in seeds})
        result.append(dict(snr_db=snr, arms=arms, accepted=sum(row["accepted"] for row in active),
            frames=len(active), candidate_ber=float(np.mean([row["ber_N_candidate"] for row in active])),
            mean_milliseconds={key: float(np.mean([row[key] for row in active])) for key in
                ("F_ms", "P_ms", "N_ms", "T_ms")},
            max_retained_norm=max(row["retained_total_norm"] for row in active)))
    return result


def run_audit(manifest_path, output_dir, *, device="cpu", smoke=False):
    """执行预先声明的开发矩阵；不接受结果驱动的命令行条件筛选。"""
    manifest_path, target = Path(manifest_path), Path(output_dir)
    spec = json.loads(manifest_path.read_text(encoding="utf-8"))
    if spec.get("schema_version") != "adaptability-development-audit-v1" or spec.get("role") != "development":
        raise ValueError("必须提供明确标为开发用途的审计清单。")
    seeds, snrs = spec["seeds"], spec["snrs_db"]
    if snrs != [0, 5, 10, 15] or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("开发审计需要完整四 SNR 与不重复 seed。")
    if spec.get("state_split") != "offline_train" or not spec.get("from_session_start") or spec.get("acquisition_count") != 1:
        raise ValueError("首轮审计必须同分布、从会话开始且仅做一次 acquisition。")
    checkpoint, config_path = ROOT / spec["checkpoint"], ROOT / spec["config"]
    if sha256(checkpoint) != spec["checkpoint_sha256"] or sha256(config_path) != spec["config_sha256"]:
        raise ValueError("开发配置或 checkpoint 指纹改变，必须先版本化清单。")
    if target.exists():
        raise ValueError("审计输出目录已存在，禁止覆盖或静默续跑。")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model, payload = _load_branch(checkpoint, config, device)
    if int(payload.get("metrics", {}).get("receiver_lookahead_len", -1)) != int(config["receiver_lookahead_len"]):
        raise ValueError("checkpoint 与审计的接收后缀协议不一致。")
    if set(seeds).intersection(item["seed"] for item in payload.get("metrics", {}).get("history", [])):
        raise ValueError("开发 seed 与 checkpoint 训练 seed 重叠。")
    frames = spec["smoke_frames"] if smoke else spec["frames"]
    if type(frames) is not int or frames < 1:
        raise ValueError("帧预算必须为正整数。")
    offsets = spec["fixed_candidate_offsets"]
    if offsets != [0, 1, 8, 32, 128] or (not smoke and frames != 128):
        raise ValueError("首轮协议固定128帧持续更新与0/1/8/32/128帧一次候选诊断。")
    settings = DeterministicSettings(**spec["deterministic_update"])
    source_names = ("evaluation/adaptability_audit.py", "training/deterministic_adaptation.py", "env/impairments.py",
        "env/extreme_delay_channel.py", "agent/unfolded_equalizer.py", "agent/neural_channel_state.py",
        "training/pilot_transfer.py", "agent/cir_estimator.py", "baseline/traditional_equalizers.py",
        "env/comm_env.py", "env/frame_structure.py", "env/experiment_config.py", "evaluation/joint_pilot_state.py",
        "online_train.py")
    source_fingerprints = {name: sha256(ROOT / name) for name in source_names}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    raw = target / "frame_metrics.jsonl"
    probe_raw = target / "fixed_candidate_metrics.jsonl"
    rows, probe_rows, started = [], [], time.perf_counter()
    base = {name: value.clone() for name, value in model.state_dict().items()}
    with raw.open("x", encoding="utf-8") as handle, probe_raw.open("x", encoding="utf-8") as probe_handle:
        for snr in snrs:
            for seed in seeds:
                neural = copy.deepcopy(model)
                adapter = DeterministicMemoryAdapter(neural, settings)
                diagnostic = ExactStateDiagnostic.from_model(model)
                env_config = build_comm_env_config(config, level="B", snr_db=snr, seed=seed,
                    max_delay=model.config.max_delay, total_pilot=config["pilot_total"],
                    reward_pilot_total=config["reward_pilot_total"], pilot_layout="prefix", state_split="offline_train")
                validate_model_dimensions(model.config, env_config)
                env = AuditEnvironment(env_config)
                start = env.reset_episode()
                cir, cfo = estimate_acquisition_cir_with_cfo(start.acquisition, model.config.max_delay,
                    cfo_limit=float(config.get("acquisition_cfo_limit", .004)))
                cir = cir.to(device)
                tail = start.initial_soft_tail.to(device)
                tracker = LegalStateTracker(cir, 10 ** (-snr / 10), config["frame_len"] - config["max_delay"],
                    float(config["rho_frame"]))
                history = []
                probes, accepted_at_source = {}, False
                for index in range(frames):
                    original = frame_to_device(env.next_frame(lookahead_len=config["receiver_lookahead_len"]), device)
                    frame = deployable_frame(original)
                    _synchronize(device)
                    tick = time.perf_counter()
                    with torch.no_grad():
                        frozen_context = make_context(frame, cir, cfo, tail, snr)
                        frozen = context_logits(model, frozen_context, visible="full_rx").detach()
                    _synchronize(device)
                    frozen_ms = 1000 * (time.perf_counter() - tick)
                    tick = time.perf_counter()
                    physical_context, block = tracker.context(model, frame, cfo, tail, snr)
                    with torch.no_grad():
                        physical = context_logits(model, physical_context, visible="full_rx").detach()
                    _synchronize(device)
                    physical_ms = 1000 * (time.perf_counter() - tick)
                    tick = time.perf_counter()
                    control, _, candidate, retained, candidate_memory = adapter.step(physical_context)
                    _synchronize(device)
                    # N 的部署代价包含共同物理跟踪；另报微调增量耗时。
                    incremental_ms = 1000 * (time.perf_counter() - tick)
                    tick = time.perf_counter()
                    truth = env.truth_for(original)
                    exact = diagnostic.diagnostic_logits(physical_context, truth)
                    _synchronize(device)
                    truth_ms = 1000 * (time.perf_counter() - tick)
                    history.append(block.design)
                    history = history[-spec["information_history_frames"]:]
                    geometry = dict(single=information_metrics(block.design), history=information_metrics(torch.cat(history)))
                    if index in spec["jacobian_frame_indices"]:
                        geometry["memory_jacobian"] = memory_jacobian_metrics(neural, physical_context, block.design)
                    # 四分支控制与留出验收均已结束，才读取原始 Data 标签评分。
                    row = dict(seed=seed, snr_db=snr, frame=index, channel_frame_index=original.frame_index,
                        rx_sha256=tensor_sha256(frame.rx_symbols), lookahead_rx_sha256=tensor_sha256(frame.lookahead_rx_symbols),
                        ber_F=_ber(frozen, original), ber_P=_ber(physical, original), ber_N=_ber(retained, original),
                        ber_N_candidate=_ber(candidate, original), ber_T=_ber(exact, original),
                        F_ms=frozen_ms, P_ms=physical_ms, N_ms=physical_ms+incremental_ms, N_incremental_ms=incremental_ms,
                        T_ms=truth_ms, information=geometry,
                        posterior_variance_trace=float(tracker.covariance.diagonal().real.sum()),
                        truth_lookahead_cir_delta_norm=float((truth.lookahead_cir-truth.cir).norm()) if truth.lookahead_cir is not None else 0.,
                        shared_physical_context_for_P_N=True, shared_tail_source="P_full_rx_soft_decisions",
                        diagnostic_only=True, **control)
                    rows.append(row)
                    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    handle.flush()
                    if index == 0:
                        accepted_at_source = control["accepted"]
                        probes = {"guarded": copy.deepcopy(neural), "raw": copy.deepcopy(neural)}
                        with torch.no_grad():
                            probes["raw"].channel_state.memory.weight.copy_(candidate_memory)
                        for branch in probes.values():
                            for parameter in branch.parameters():
                                parameter.requires_grad_(False)
                    if index in offsets:
                        probe = fixed_candidate_probe(probes, physical_context, physical, original,
                            seed=seed, snr=snr, accepted=accepted_at_source)
                        probe_rows.append(probe)
                        probe_handle.write(json.dumps(probe, ensure_ascii=False, allow_nan=False) + "\n")
                        probe_handle.flush()
                    # 所有分支共用合法 P 输出的 soft tail，避免 N 改变物理跟踪而混淆归因。
                    with torch.no_grad():
                        values = torch.tanh(physical[-model.config.max_delay:] / 2.)
                        tail = torch.complex(values, torch.zeros_like(values)).detach()
                if not smoke:
                    # 多接收一帧仅用于首帧候选的128偏移诊断，不追加神经更新、不混入主128帧平均。
                    original = frame_to_device(env.next_frame(lookahead_len=config["receiver_lookahead_len"]), device)
                    context, _ = tracker.context(model, deployable_frame(original), cfo, tail, snr)
                    with torch.no_grad():
                        physical = context_logits(model, context, visible="full_rx")
                    probe = fixed_candidate_probe(probes, context, physical, original,
                        seed=seed, snr=snr, accepted=accepted_at_source)
                    probe_rows.append(probe)
                    probe_handle.write(json.dumps(probe, ensure_ascii=False, allow_nan=False) + "\n")
                    probe_handle.flush()
                if any(not torch.equal(base[name], value) for name, value in model.state_dict().items()):
                    raise AssertionError("审计不得修改同源冻结网络。")
                print(json.dumps(dict(seed=seed, snr_db=snr, frames_complete=frames), ensure_ascii=False), flush=True)
    if any(sha256(ROOT / name) != value for name, value in source_fingerprints.items()):
        raise RuntimeError("审计期间源文件改变，结果不予汇总；保留原始日志供核查。")
    summary = dict(schema_version="adaptability-development-result-v1", role="development", smoke=smoke,
        device=str(device), torch_version=torch.__version__, torch_threads=torch.get_num_threads(),
        cuda_device_name=torch.cuda.get_device_name(device) if torch.device(device).type == "cuda" else None,
        core_confirmed=False, confirmation_ready=False, uses_rl=False,
        manifest_sha256=sha256(manifest_path), checkpoint_sha256=sha256(checkpoint), config_sha256=sha256(config_path),
        source_revision=revision, source_sha256=source_fingerprints, seeds=seeds, snrs_db=snrs,
        frames_per_seed_snr=frames, records=len(rows), per_snr=summarize(rows, seeds, snrs),
        raw_sha256=sha256(raw), fixed_candidate_records=len(probe_rows), fixed_candidate_raw_sha256=sha256(probe_raw),
        fixed_candidate_offsets_observed=sorted({row["offset"] for row in probe_rows}),
        frozen_weights_unchanged=True, wall_seconds=time.perf_counter()-started,
        truth_limitations="T只替换同步和当前CIR，不使用真实tail；现有算子仍以当前CIR近似后缀CIR，不是最优界。",
        timing_scope="GPU同步；F含条件与检测，P含递推与检测，N含P耗时及重复前向/梯度/守门；不含环境/acquisition/信息几何/评分。")
    (target / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return summary
