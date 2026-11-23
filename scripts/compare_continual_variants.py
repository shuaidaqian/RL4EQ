# -*- coding: utf-8 -*-
"""用预先声明的候选清单作同波形开发比较，不覆盖旧实验。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from evaluation.continual_variants import (
    ForwardCounter,
    KnownAdaptRealNeuralBranch,
    KnownAdaptRealPhysicsBranch,
    PhysicsBranch,
    validate_variants,
)
from evaluation.channel_dynamics import load_ar1_dynamics
from evaluation.pilot_transfer import _load_branch
from evaluation.receiver_diagnostics import (
    LONG_DIAGNOSTIC_FIELDS, RECEIVER_SCORE_FIELDS, long_receiver_diagnostics, receiver_diagnostics,
)
from evaluation.seed_registry import reserve_seeds
from scripts.develop_continual_receiver import fingerprint, summarize, synchronize
from training.continual_receiver import (
    ContinualReceiver, acquisition_transition_intervals, estimate_acquisition, receiver_frame,
)
from training.continual_accel import BatchedForward, GraphBatchedForward
from training.continual_offline import InputRecorder
from training.pilot_transfer import context_model_inputs, frame_to_device


STREAM_ROW_FIELDS = (
    'seed', 'snr_db', 'frame_index', 'arm', 'errors', 'bits', 'elapsed_ms', 'rx_sha256',
    'prior_covariance_trace', 'decode_covariance_trace', 'posterior_covariance_trace',
    'pilot_observations', 'pilot_innovation_mse', 'pilot_innovation_variance_mean',
    'noise_variance_used', 'pilot_updates', 'data_updates', 'neural_forwards', 'model_calls',
    'true_cir_norm_diagnostic',
)
SOLVER_DIAGNOSTIC_FIELDS = (
    'physics_solver_iterations', 'physics_solver_relative_residual', 'physics_solver_rhs_norm',
)
MINIMAL_STREAM_FIELDS = (
    'seed', 'snr_db', 'frame_index', 'arm', 'errors', 'bits', 'elapsed_ms', 'rx_sha256',
    'true_cir_norm_diagnostic', 'model_calls', 'neural_forwards',
)


def validate_solver_diagnostics(row, variant):
    """求解诊断必须与预登记开关和实际迭代次数一致。"""

    present = [key in row for key in SOLVER_DIAGNOSTIC_FIELDS]
    enabled = variant.get('physics_solver_diagnostics', False)
    if (enabled and not all(present)) or (not enabled and any(present)):
        raise ValueError('物理求解诊断与预登记开关不一致或字段缺失。')
    if not enabled:
        return ()
    iterations = row['physics_solver_iterations']
    if (type(iterations) is not int or iterations <= 0 or
            iterations != variant.get('physics_solver_iterations', iterations)):
        raise ValueError('物理求解诊断中的迭代次数不匹配。')
    for key in SOLVER_DIAGNOSTIC_FIELDS[1:]:
        if type(row[key]) not in {int, float} or not math.isfinite(row[key]) or row[key] < 0:
            raise ValueError('物理求解残差与右端范数须为非负有限数。')
    return SOLVER_DIAGNOSTIC_FIELDS


def compact_stream_row(row, *, schema_version=2):
    """新流式格式明确保留合法观测诊断；缺字段时拒绝静默丢弃。"""

    if schema_version not in {2, 3, 4}:
        raise ValueError('未知流式诊断版本。')
    if schema_version == 4:
        fields = MINIMAL_STREAM_FIELDS
    else:
        fields = STREAM_ROW_FIELDS + (RECEIVER_SCORE_FIELDS + LONG_DIAGNOSTIC_FIELDS
                                      if schema_version == 3 else ())
    missing = [key for key in fields if key not in row]
    if missing:
        raise ValueError(f'流式诊断记录缺少必要字段：{missing}')
    compact = {key: row[key] for key in fields}
    if 'shared_state_from' in row:
        compact['shared_state_from'] = row['shared_state_from']
    solver_fields = SOLVER_DIAGNOSTIC_FIELDS
    if any(key in row for key in solver_fields):
        if any(key not in row for key in solver_fields):
            raise ValueError('物理求解诊断字段必须完整保存。')
        compact.update({key: row[key] for key in solver_fields})
    return compact


def inspect_stream_for_resume(rows_output, *, snrs, seeds, arms, frames):
    """检查已有流文件，只允许完整会话加一个连续前缀会话。"""
    session_order = {(snr, seed): index
                     for index, (snr, seed) in enumerate((snr, seed) for snr in snrs for seed in seeds)}
    completed = set()
    current_key = None
    current_rows = None
    partial_key = None
    partial_rows = None
    total_rows = 0
    last_order = -1

    def finish_session(key, rows_by_arm):
        nonlocal partial_key, partial_rows
        if key is None:
            return
        sets = {arm: set(rows_by_arm[arm]) for arm in arms}
        if all(values == set(range(frames)) for values in sets.values()):
            complete = True
        else:
            lengths = {len(values) for values in sets.values()}
            count = next(iter(lengths)) if len(lengths) == 1 else -1
            complete = False
            if count <= 0 or count >= frames or any(values != set(range(count)) for values in sets.values()):
                raise ValueError(f"续跑流文件中的会话不是完整数据或共同连续前缀：{key}")
            partial_key = key
            partial_rows = rows_by_arm
            for frame_index in range(count):
                hashes = {rows_by_arm[arm][frame_index].get('rx_sha256') for arm in arms}
                if len(hashes) != 1 or None in hashes:
                    raise ValueError(f"续跑流文件已有部分会话的配对波形不一致：{key} 帧 {frame_index}")
        if complete:
            for frame_index in range(frames):
                hashes = {rows_by_arm[arm][frame_index].get('rx_sha256') for arm in arms}
                if len(hashes) != 1 or None in hashes:
                    raise ValueError(f"续跑流文件已有完整会话的配对波形不一致：{key} 帧 {frame_index}")
            completed.add(key)

    with Path(rows_output).open('r', encoding='utf8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if 'frame_index' not in row:
                raise ValueError(f"逐帧流第 {line_number} 行缺少帧索引。")
            key = (row.get('snr_db'), row.get('seed'))
            if key not in session_order or row.get('arm') not in arms:
                raise ValueError(f"逐帧流第 {line_number} 行超出预登记协议。")
            order = session_order[key]
            if key != current_key:
                if order != last_order + 1:
                    raise ValueError(f"逐帧流会话顺序异常或会话缺失：{key}")
                finish_session(current_key, current_rows)
                if partial_key is not None:
                    raise ValueError("不完整会话之后又出现了后续会话，拒绝自动续跑。")
                current_key = key
                current_rows = {arm: {} for arm in arms}
                last_order = order
            frame_index = row['frame_index']
            if not isinstance(frame_index, int) or not 0 <= frame_index < frames:
                raise ValueError(f"逐帧流第 {line_number} 行帧索引越界。")
            arm_rows = current_rows[row['arm']]
            if frame_index in arm_rows:
                raise ValueError(f"逐帧流出现重复记录：{key} {row['arm']} 帧 {frame_index}")
            arm_rows[frame_index] = row
            total_rows += 1
    finish_session(current_key, current_rows)
    if partial_key is None and current_key is not None and current_key not in completed:
        raise ValueError("无法分类续跑流文件的末尾会话。")
    return completed, partial_key, partial_rows, total_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/eme_long_memory_v2_online_dd_tail_v1.json')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--snrs', nargs='+', type=int, default=[0, 5, 10, 15])
    parser.add_argument('--state-split', default='offline_train')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stream-rows', action='store_true',
                        help='将逐帧结果追加到 JSONL，主 JSON 只保存协议与状态。')
    parser.add_argument('--long-diagnostics', action='store_true',
                        help='接收完成后保存实际解码输入、状态与同步的事后标量评分；流式格式为3。')
    parser.add_argument('--checkpoint-interval', type=int, default=8,
                        help='按多少帧原子保存一次；流式长会话可设为较大值。')
    parser.add_argument('--resume', action='store_true',
                        help='续跑已登记且中断的流式实验；验证已有前缀后只追加新帧。')
    parser.add_argument('--accel', choices=['none', 'batch', 'graph'], default='none',
                        help='两臂合并前向（batch）或再叠加 CUDA Graph（graph）；默认关闭。')
    parser.add_argument('--purpose', choices=['development', 'confirmation'], default='development',
                        help='seed registry 中的实验角色；独立确认必须是 confirmation。')
    parser.add_argument('--minimal-diagnostics', action='store_true',
                        help='只记录确认分析所需字段（schema 4），跳过逐帧浮点诊断同步。')
    args = parser.parse_args()
    stream_schema = 4 if args.minimal_diagnostics else (3 if args.long_diagnostics else 2)
    if args.minimal_diagnostics and (args.long_diagnostics or not args.stream_rows):
        parser.error('最小诊断只支持流式记录，且不能与长诊断同时使用。')
    if args.frames < 1 or args.checkpoint_interval < 1 or not args.snrs or len(set(args.snrs)) != len(args.snrs):
        parser.error('需要正帧数和互异的 SNR。')
    output = Path(args.output)
    rows_output = output.with_suffix('.jsonl')
    if args.resume:
        if not args.stream_rows or not output.exists() or not rows_output.exists():
            raise ValueError('续跑要求流式模式且主 JSON、JSONL 文件均存在。')
    elif output.exists() or (args.stream_rows and rows_output.exists()):
        raise ValueError('禁止覆盖或隐式重跑旧实验。')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf8'))
    if args.stream_rows and manifest.get('expected_stream_schema_version', stream_schema) != stream_schema:
        raise ValueError('命令行流式诊断版本与预登记协议不一致。')
    variants = manifest.get('variants', [])
    initial_intervals = acquisition_transition_intervals(
        config['frame_duration_seconds'], config.get('acquisition_to_data_gap_seconds', 0.0))
    for row in variants:
        # 显式写入实际采用的时序；0 仅用于另行登记的历史初始化对照。
        row.setdefault('initial_prediction_intervals', initial_intervals)
        row.setdefault('current_pilot_prior', 'acquisition')
        if row.get('state_model', 'random_walk') == 'ar1' and row.get('dynamics_file'):
            dynamics_path = Path(row['dynamics_file'])
            if not dynamics_path.is_absolute():
                dynamics_path = ROOT / dynamics_path
            expected_hash = row.get('dynamics_file_sha256')
            if not expected_hash:
                raise ValueError(f"AR(1) 候选必须登记动态文件指纹：{dynamics_path}")
            dynamics = load_ar1_dynamics(
                dynamics_path, expected_sha256=expected_hash,
                tap_count=config['max_delay'] + 1,
            )
            row['state_rho'] = dynamics['rho_estimate']
            row['stationary_power_by_tap'] = dynamics['stationary_power_by_tap']
    variants = validate_variants(manifest)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    cache, models, recorders = {}, {}, {}
    for row in variants:
        checkpoint = str(Path(row['checkpoint']).resolve())
        if checkpoint not in cache:
            cache[checkpoint], _ = _load_branch(
                Path(checkpoint), config, device,
                allow_real_bpsk_known_adapt_override=(
                    row.get('decoder') == 'known_adapt_real_neural'
                ),
            )
        model = cache[checkpoint]
        decoder = row.get('decoder', 'neural')
        if decoder == 'neural':
            branch = model
        elif decoder == 'known_adapt_real_neural':
            branch = KnownAdaptRealNeuralBranch(model)
        elif decoder == 'known_adapt_real_physics':
            branch = KnownAdaptRealPhysicsBranch(model,
                solver_iterations=row.get('physics_solver_iterations'),
                solver_diagnostics=row.get('physics_solver_diagnostics', False))
        else:
            branch = PhysicsBranch(model)
        recorders[row['name']] = InputRecorder(branch)
        models[row['name']] = ForwardCounter(recorders[row['name']])
    accel_model, accel_decoder, accelerated_backend = None, None, None
    if args.accel != 'none':
        if not args.stream_rows or args.long_diagnostics:
            raise ValueError('加速模式只支持 schema 2 流式记录，不支持长诊断。')
        if len(variants) < 2:
            raise ValueError('加速模式至少需要两条候选臂。')
        if any(row.get('decoder', 'neural') != 'neural' or row.get('context_source')
               or row.get('physics_solver_diagnostics') for row in variants):
            raise ValueError('加速模式只支持 neural 解码臂，且不支持共享上下文或求解诊断。')
        if len(cache) != 1:
            raise ValueError('加速模式要求两条臂共享同一 checkpoint。')
        backends = {row.get('operator_backend', os.environ.get('RL4EQ_LINEAR_OPERATOR_BACKEND', 'direct'))
                    for row in variants}
        if len(backends) != 1:
            raise ValueError('加速模式要求两条臂使用同一线性算子后端。')
        accelerated_backend = backends.pop()
        accel_model = next(iter(cache.values()))
        accel_model.eval()
        if args.accel == 'graph' and accel_model.config.enable_channel_state_branch:
            raise ValueError('CUDA Graph 不支持按帧变化的 frame_index 输入。')
        accel_decoder = (BatchedForward(accel_model) if args.accel == 'batch'
                         else GraphBatchedForward(accel_model))
    protocol = dict(vars(args), variants=variants, arms=[row['name'] for row in variants],
                    acceleration=args.accel, shared_forward=args.accel != 'none',
                    minimal_diagnostics=args.minimal_diagnostics,
                    stream_rows_schema_version=stream_schema if args.stream_rows else None,
                    posthoc_diagnostic='actual_decoder_input_frozen_phase_replay_no_feedback' if args.long_diagnostics else None,
                    pilot_innovation_diagnostic='clean_adapt_pre_update_phase_fitted_not_chi_square',
                    pairs=manifest.get('pairs', []), role=args.purpose, noise_source='acquisition_residual',
                    statistical_unit='session', timing='current_pilot_then_decode',
                    linear_operator_backend=os.environ.get('RL4EQ_LINEAR_OPERATOR_BACKEND', 'direct'),
                    torch_version=torch.__version__, config_sha256=fingerprint(args.config),
                    manifest_sha256=fingerprint(args.manifest),
                    checkpoint_sha256={p:fingerprint(p) for p in cache},
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    sources={name:fingerprint(ROOT/name) for name in [
                        'training/continual_receiver.py', 'training/continual_phase.py',
                        'evaluation/continual_variants.py', 'scripts/compare_continual_variants.py',
                        'evaluation/channel_dynamics.py',
                        'evaluation/receiver_diagnostics.py',
                        'agent/unfolded_equalizer.py', 'env/linear_operator.py',
                        'env/comm_env.py', 'env/extreme_delay_channel.py', 'env/impairments.py',
                        'env/eme_channel_profiles.py', 'training/continual_offline.py',
                        'training/continual_accel.py', 'training/pilot_transfer.py']})
    completed_sessions, partial_session, partial_rows = set(), None, None
    previous_acquisitions = {}
    if args.resume:
        registry_path = ROOT / 'docs/protocols/continual_v2_seed_registry.json'
        registry = json.loads(registry_path.read_text(encoding='utf8'))
        registration = next((row for row in registry['experiments']
                             if row.get('experiment') == args.experiment), None)
        if (registration is None or registration.get('status') != 'running'
                or registration.get('purpose') != args.purpose
                or registration.get('seeds') != args.seeds):
            raise ValueError('续跑只允许使用完全匹配且状态为 running 的开发登记。')
        metadata_candidates = [output.with_suffix('.tmp'), output]
        previous = None
        for candidate in metadata_candidates:
            if candidate.exists():
                loaded = json.loads(candidate.read_text(encoding='utf8'))
                if candidate == output.with_suffix('.tmp') and loaded.get('status') != 'failed':
                    continue
                previous = loaded
                break
        if previous is None or previous.get('status') not in {'failed', 'running'}:
            raise ValueError('找不到可安全续跑的失败或运行中主元数据。')
        previous_protocol = previous.get('protocol', {})
        if registration.get('details') != previous_protocol:
            raise ValueError('种子登记与结果文件协议不一致，拒绝续跑。')
        compatible_fields = (
            'config', 'manifest', 'seeds', 'frames', 'snrs', 'state_split', 'experiment',
            'output', 'device', 'stream_rows', 'checkpoint_interval', 'config_sha256',
            'manifest_sha256', 'checkpoint_sha256', 'variants', 'arms', 'pairs',
            'role', 'noise_source', 'statistical_unit', 'timing', 'linear_operator_backend',
            'stream_rows_schema_version', 'pilot_innovation_diagnostic',
            'long_diagnostics', 'posthoc_diagnostic', 'accel', 'purpose',
            'minimal_diagnostics',
        )
        for field in compatible_fields:
            if previous_protocol.get(field) != protocol.get(field):
                raise ValueError(f'续跑参数与原协议不一致：{field}')
        for name, digest in previous_protocol.get('sources', {}).items():
            if name == 'scripts/compare_continual_variants.py':
                continue
            if fingerprint(ROOT / name) != digest:
                raise ValueError(f'续跑时检测到算法源文件已变化：{name}')
        completed_sessions, partial_session, partial_rows, streamed_rows = inspect_stream_for_resume(
            rows_output, snrs=args.snrs, seeds=args.seeds,
            arms=[row['name'] for row in variants], frames=args.frames)
        if previous.get('row_count') != streamed_rows:
            raise ValueError('失败元数据的 row_count 与逐帧流记录数不一致，拒绝续跑。')
        result = previous
        result['protocol'] = previous_protocol
        result['rows'] = []
        result['status'] = 'running'
        result.pop('failure', None)
        result.setdefault('recovery_events', []).append({
            'unix_time': time.time(),
            'resume_git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'resume_script_sha256': fingerprint(ROOT / 'scripts/compare_continual_variants.py'),
            'replayed_session': list(partial_session) if partial_session else None,
            'replayed_through_frame': max(max(partial_rows[arm]) for arm in partial_rows)
                if partial_rows else None,
            'reason': previous.get('failure'),
        })
        protocol = previous_protocol
        previous_acquisitions = {
            (row['snr_db'], row['seed']): row for row in result.get('acquisitions', [])
        }
    else:
        reserve_seeds(ROOT, experiment=args.experiment, seeds=args.seeds,
                      purpose=args.purpose, details=protocol)
        output.parent.mkdir(parents=True, exist_ok=True)
        result = dict(protocol=protocol, acquisitions=[], rows=[], status='running')
        streamed_rows = 0
    session_errors, session_bits = {}, {}

    def save():
        nonlocal streamed_rows
        if args.stream_rows:
            if result['rows']:
                compacted_rows = [compact_stream_row(row, schema_version=stream_schema) for row in result['rows']]
                with rows_output.open('a', encoding='utf8', newline='\n') as stream:
                    for compact in compacted_rows:
                        stream.write(json.dumps(compact, ensure_ascii=False, separators=(',', ':'))+'\n')
                    stream.flush()
                streamed_rows += len(result['rows'])
                result['rows'] = []
            result['rows_file'] = rows_output.name
            result['row_count'] = streamed_rows
        else:
            result['summary'] = summarize(result['rows'])
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        for attempt in range(20):
            try:
                temporary.replace(output)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(min(0.25 * (attempt + 1), 2.0))

    save()
    try:
        for snr in args.snrs:
            for seed in args.seeds:
                session_id = (snr, seed)
                if session_id in completed_sessions:
                    continue
                env = CommunicationEnvironment(build_comm_env_config(config, level='B', snr_db=snr, seed=seed,
                    max_delay=config['max_delay'], total_pilot=config['pilot_total'],
                    reward_pilot_total=config['reward_pilot_total'], pilot_layout='prefix', state_split=args.state_split))
                start = env.reset_episode()
                acquisition = estimate_acquisition(start.acquisition.tx_symbols, start.acquisition.rx_symbols,
                    config['max_delay'], cfo_limit=config['acquisition_cfo_limit'])
                acquisition_record = dict(seed=seed, snr_db=snr,
                    noise_estimate=acquisition.noise_variance, noise_truth_diagnostic=10**(-snr/10),
                    cfo_estimate=acquisition.cfo, cir_norm=float(acquisition.mean.norm()))
                prior_acquisition = previous_acquisitions.get(session_id)
                if prior_acquisition is None:
                    result['acquisitions'].append(acquisition_record)
                else:
                    for field in ('noise_estimate', 'noise_truth_diagnostic', 'cfo_estimate', 'cir_norm'):
                        if not math.isclose(float(prior_acquisition[field]), float(acquisition_record[field]),
                                            rel_tol=1e-9, abs_tol=1e-9):
                            raise ValueError(f'重放会话的 acquisition 估计不一致：{session_id} {field}')
                acquisition = replace(acquisition, mean=acquisition.mean.to(device),
                                      covariance=acquisition.covariance.to(device))
                receivers = {row['name']:ContinualReceiver(acquisition, start.initial_soft_tail.to(device),
                    arm=row['arm'], process_noise=row.get('process_noise', 1e-4),
                    phase_estimator=row.get('phase_estimator', 'unwrap_median'),
                    phase_reference_backend=row.get('phase_reference_backend', 'vectorized'),
                    state_model=row.get('state_model', 'random_walk'),
                    state_rho=row.get('state_rho'),
                    stationary_power=row.get('stationary_power_by_tap'),
                    initial_prediction_intervals=row['initial_prediction_intervals'],
                    current_pilot_prior=row['current_pilot_prior']) for row in variants}
                for _ in range(args.frames):
                    original = frame_to_device(env.next_frame(), device)
                    visible = receiver_frame(original)
                    rx_hash = hashlib.sha256(original.rx_symbols.cpu().numpy().tobytes()).hexdigest()
                    frame_logits, frame_diagnostics = {}, {}
                    accelerated = None
                    if accel_decoder is not None:
                        prepared_steps, calls = {}, []
                        previous_backend = os.environ.get('RL4EQ_LINEAR_OPERATOR_BACKEND')
                        os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = accelerated_backend
                        try:
                            for row in variants:
                                prepared = receivers[row['name']].prepare(
                                    visible, diagnostics=not args.minimal_diagnostics)
                                prepared_steps[row['name']] = prepared
                                calls.append(context_model_inputs(
                                    accel_model, prepared.context, visible='data'))
                            synchronize(device)
                            began = time.perf_counter()
                            with torch.no_grad():
                                batch_logits = accel_decoder(calls)
                            synchronize(device)
                            shared_elapsed = (time.perf_counter()-began)*1000
                        finally:
                            if previous_backend is None:
                                os.environ.pop('RL4EQ_LINEAR_OPERATOR_BACKEND', None)
                            else:
                                os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = previous_backend
                        accelerated = (prepared_steps, batch_logits, shared_elapsed)
                    for index, row in enumerate(variants):
                        name = row['name']
                        decoder = models[name]
                        if accelerated is not None:
                            prepared_steps, batch_logits, shared_elapsed = accelerated
                            logits, diagnostics = receivers[name].finish(
                                prepared_steps[name], batch_logits[index],
                                diagnostics=not args.minimal_diagnostics)
                            elapsed = shared_elapsed
                            model_calls = 1
                        else:
                            before = decoder.calls
                            synchronize(device)
                            began = time.perf_counter()
                            previous_backend = os.environ.get('RL4EQ_LINEAR_OPERATOR_BACKEND')
                            os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = row.get('operator_backend', protocol['linear_operator_backend'])
                            try:
                                if row.get('context_source'):
                                    source = row['context_source']
                                    actual_args, actual_kwargs = recorders[source].last_inputs
                                    with torch.no_grad():
                                        batch_logits, _ = decoder(*actual_args, **actual_kwargs)
                                    logits = batch_logits.squeeze(0)
                                    diagnostics = dict(frame_diagnostics[source])
                                    diagnostics.update(pilot_updates=0, data_updates=0, shared_state_from=source)
                                else:
                                    logits, diagnostics = receivers[name].step(
                                        decoder, visible,
                                        diagnostics=not args.minimal_diagnostics)
                            finally:
                                if previous_backend is None:
                                    os.environ.pop('RL4EQ_LINEAR_OPERATOR_BACKEND', None)
                                else:
                                    os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = previous_backend
                            synchronize(device)
                            elapsed = (time.perf_counter()-began)*1000
                            model_calls = decoder.calls-before
                        if accelerated is None and row.get('physics_solver_diagnostics', False):
                            solver = recorders[name].model.last_solver_diagnostics
                            diagnostics.update(physics_solver_iterations=solver['iterations'],
                                physics_solver_relative_residual=float(solver['relative_residual'].item()),
                                physics_solver_rhs_norm=float(solver['rhs_norm'].item()))
                        frame_diagnostics[name] = dict(diagnostics)
                        # 接收已结束，以下全部为事后评分，不回流任何更新。
                        mask = original.data_mask
                        errors = int(((logits[mask]>=0) != original.bits[mask].bool()).sum())
                        session_key = (snr, seed, name)
                        session_errors[session_key] = session_errors.get(session_key, 0) + errors
                        session_bits[session_key] = session_bits.get(session_key, 0) + int(mask.sum())
                        if args.minimal_diagnostics:
                            result['rows'].append(dict(
                                seed=seed, snr_db=snr, frame_index=original.frame_index,
                                arm=name, errors=errors, bits=int(mask.sum()),
                                elapsed_ms=elapsed, model_calls=model_calls,
                                rx_sha256=rx_hash, neural_forwards=1,
                                true_cir_norm_diagnostic=float(original.true_cir.flatten().norm())))
                            continue
                        estimate = diagnostics.pop('decode_cir')
                        truth = original.true_cir.flatten()
                        scoring = receiver_diagnostics(logits, original.bits, mask, estimate, truth)
                        if args.long_diagnostics:
                            scoring.update(long_receiver_diagnostics(
                                recorders[name].model, recorders[name].last_inputs, original,
                                env.channel.last_phase_used(), env.state_metadata()['cfo_cycles_per_symbol'],
                                10**(-snr/10)))
                        if accelerated is None:
                            diagnostics['neural_forwards'] = (decoder.calls-before)*int(
                                row.get('decoder', 'neural') in {'neural', 'known_adapt_real_neural'}
                            )
                        paired_logits = {}
                        for baseline, online in protocol['pairs']:
                            if online == name and baseline in frame_logits:
                                other = frame_logits[baseline]
                                paired_logits[baseline] = dict(
                                    max_abs=float((logits-other).abs().max()),
                                    data_disagreements=int(((logits[mask]>=0)!=(other[mask]>=0)).sum()))
                        frame_logits[name] = logits.detach()
                        result['rows'].append(dict(seed=seed, snr_db=snr, frame_index=original.frame_index,
                            arm=name, errors=errors, bits=int(mask.sum()), ber=errors/int(mask.sum()),
                            elapsed_ms=elapsed, model_calls=model_calls, **scoring,
                            paired_logits_diagnostic=paired_logits,
                            noise_variance_used=acquisition.noise_variance, rx_sha256=rx_hash, **diagnostics))
                    if partial_session == session_id and original.frame_index <= max(
                            max(partial_rows[arm]) for arm in partial_rows):
                        replayed_rows = result['rows'][-len(variants):]
                        for replayed in replayed_rows:
                            stored = partial_rows[replayed['arm']].get(original.frame_index)
                            if stored is None or any(replayed[field] != stored[field]
                                                     for field in ('errors', 'bits', 'rx_sha256')):
                                del result['rows'][-len(variants):]
                                raise ValueError(
                                    f'重放前缀与已保存帧不一致：{session_id} '
                                    f"{original.frame_index} {replayed['arm']}"
                                )
                        del result['rows'][-len(variants):]
                    if ((original.frame_index + 1) % args.checkpoint_interval == 0
                            or original.frame_index + 1 == args.frames):
                        save()
                save()
                print(f'候选比较 SNR={snr} seed={seed} '+ ' '.join(
                    f"{name}={session_errors[(snr, seed, name)]/session_bits[(snr, seed, name)]:.6f}"
                    for name in receivers), flush=True)
        result['status'] = 'completed'
    except BaseException as exc:
        result['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        result['failure'] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    save()


if __name__ == '__main__':
    main()
