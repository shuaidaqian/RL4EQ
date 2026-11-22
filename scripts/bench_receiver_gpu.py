# -*- coding: utf-8 -*-
"""只读吞吐基准：测量连续接收前向的批量化/编译/CUDA Graph 收益。

不写实验结论、不做 BER 评分、不登记 seed；仅用于选择下一轮实现方案。
默认跑 ``--mode step`` 的真实两臂逐帧路径；``--mode forward`` 使用首帧实际
输入做解码器微基准（batch=1 vs batch=2，可选 torch.compile / CUDA Graph）。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from agent.cir_estimator import CIRCondition
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from evaluation.channel_dynamics import load_ar1_dynamics
from evaluation.pilot_transfer import _load_branch
from training.continual_offline import InputRecorder
from training.continual_receiver import (
    ContinualReceiver, acquisition_transition_intervals, estimate_acquisition, receiver_frame,
)
from training.pilot_transfer import frame_to_device


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def timed(fn, reps, device, warmup=3):
    for _ in range(warmup):
        fn()
    synchronize(device)
    samples = []
    for _ in range(reps):
        synchronize(device)
        began = time.perf_counter()
        fn()
        synchronize(device)
        samples.append((time.perf_counter() - began) * 1000.0)
    return dict(mean_ms=statistics.mean(samples), median_ms=statistics.median(samples),
                min_ms=min(samples), max_ms=max(samples), reps=reps)


def _batch_join(*values):
    """沿 batch 维合并；一维张量没有 batch 维，需用 stack 而不是 cat。"""
    if len(values) == 1:
        return values[0].clone()
    if values[0].ndim == 1:
        return torch.stack(values, dim=0)
    return torch.cat(values, dim=0)


def _stack_condition(condition, copies):
    if copies == 1:
        return condition
    return replace(condition, **{
        name: _batch_join(*([value] * copies))
        for name, value in condition.__dict__.items()
        if torch.is_tensor(value)
    })


def _stack_call(args, kwargs, copies):
    """把一次实际前向的输入沿 batch 维复制 ``copies`` 份。"""
    stacked_args = []
    for value in args:
        if torch.is_tensor(value):
            stacked_args.append(_batch_join(*([value] * copies)))
        elif isinstance(value, CIRCondition):
            stacked_args.append(_stack_condition(value, copies))
        else:
            stacked_args.append(value)
    stacked_kwargs = {
        key: (_batch_join(*([value] * copies)) if torch.is_tensor(value) else value)
        for key, value in kwargs.items()
    }
    return tuple(stacked_args), stacked_kwargs


def _merge_calls(calls):
    """把两条臂各自的前向输入拼成一次 batch=2 调用。"""
    args_a, kwargs_a = calls[0]
    args_b, kwargs_b = calls[1]
    merged_args = []
    for a, b in zip(args_a, args_b):
        if torch.is_tensor(a):
            merged_args.append(_batch_join(a, b))
        elif isinstance(a, CIRCondition):
            merged_args.append(replace(a, **{
                name: _batch_join(getattr(a, name), getattr(b, name))
                for name in a.__dict__ if torch.is_tensor(getattr(a, name))
            }))
        else:
            merged_args.append(a)
    merged_kwargs = {}
    for key, value in kwargs_a.items():
        other = kwargs_b[key]
        merged_kwargs[key] = _batch_join(value, other) if torch.is_tensor(value) else value
    return tuple(merged_args), merged_kwargs


def build_receivers(config, manifest, device, seed, snr):
    env = CommunicationEnvironment(build_comm_env_config(
        config, level='B', snr_db=snr, seed=seed, max_delay=config['max_delay'],
        total_pilot=config['pilot_total'], reward_pilot_total=config['reward_pilot_total'],
        pilot_layout='prefix', state_split='offline_train'))
    start = env.reset_episode()
    acquisition = estimate_acquisition(start.acquisition.tx_symbols, start.acquisition.rx_symbols,
                                       config['max_delay'], cfo_limit=config['acquisition_cfo_limit'])
    acquisition = replace(acquisition, mean=acquisition.mean.to(device),
                          covariance=acquisition.covariance.to(device))
    tail = start.initial_soft_tail.to(device)
    initial_intervals = acquisition_transition_intervals(
        config['frame_duration_seconds'], config.get('acquisition_to_data_gap_seconds', 0.0))
    receivers = {}
    for row in manifest['variants']:
        dynamics = {}
        if row.get('state_model') == 'ar1':
            dynamics_path = Path(row['dynamics_file'])
            if not dynamics_path.is_absolute():
                dynamics_path = ROOT / dynamics_path
            dynamics = load_ar1_dynamics(
                dynamics_path, expected_sha256=row['dynamics_file_sha256'],
                tap_count=config['max_delay'] + 1)
        receivers[row['name']] = ContinualReceiver(
            acquisition, tail, arm=row['arm'], process_noise=row.get('process_noise', 0.0),
            phase_estimator=row.get('phase_estimator', 'circular_all'),
            phase_reference_backend=row.get('phase_reference_backend', 'vectorized'),
            state_model=row.get('state_model', 'random_walk'),
            state_rho=dynamics.get('rho_estimate'),
            stationary_power=dynamics.get('stationary_power_by_tap'),
            initial_prediction_intervals=row.get('initial_prediction_intervals', initial_intervals),
            current_pilot_prior=row.get('current_pilot_prior', 'acquisition'))
    return env, receivers


def _checkpoint_path(manifest):
    path = Path(manifest['variants'][0]['checkpoint'])
    return path if path.is_absolute() else ROOT / path


def bench_step(config, manifest, device, seed, snr, frames):
    env, receivers = build_receivers(config, manifest, device, seed, snr)
    model, _ = _load_branch(_checkpoint_path(manifest), config, device)
    model.eval()
    arms = list(receivers)
    per_arm = {name: [] for name in arms}
    frame_times = []
    for _ in range(frames):
        frame = receiver_frame(frame_to_device(env.next_frame(), device))
        started = time.perf_counter()
        for name in arms:
            synchronize(device)
            before = time.perf_counter()
            receivers[name].step(model, frame)
            synchronize(device)
            per_arm[name].append((time.perf_counter() - before) * 1000.0)
        frame_times.append((time.perf_counter() - started) * 1000.0)
    return {
        'frames': frames,
        'per_arm': {name: dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values))
                    for name, values in per_arm.items()},
        'frame_mean_ms': statistics.mean(frame_times),
        'frame_median_ms': statistics.median(frame_times),
        'frames_per_second': 1000.0 / statistics.mean(frame_times),
    }


def bench_forward(config, manifest, device, seed, snr, reps):
    env, receivers = build_receivers(config, manifest, device, seed, snr)
    model, _ = _load_branch(_checkpoint_path(manifest), config, device)
    model.eval()
    arms = list(receivers)
    if len(arms) != 2:
        raise ValueError('forward 基准需要恰好两条候选臂。')
    recorder = InputRecorder(model)
    frame = receiver_frame(frame_to_device(env.next_frame(), device))
    for name in arms:
        receivers[name].step(recorder, frame)
    # 每条臂各自的首帧实际输入（batch=1）。
    per_arm_calls = {}
    for name in arms:
        recorder.last_inputs = None
        receivers[name].step(recorder, frame)
        per_arm_calls[name] = recorder.last_inputs
    calls = [per_arm_calls[name] for name in arms]

    result = {}
    with torch.no_grad():
        for name, (args, kwargs) in per_arm_calls.items():
            result[f'single_{name}'] = timed(lambda a=args, k=kwargs: model(*a, **k), reps, device)
        batch_args, batch_kwargs = _merge_calls(calls)
        result['batch2'] = timed(lambda: model(*batch_args, **batch_kwargs), reps, device)

        # 数值一致性：逐臂 batch=1 与合并 batch=2 的 logits 最大差。
        single_logits = [model(*per_arm_calls[name][0], **per_arm_calls[name][1])[0]
                         for name in arms]
        merged_logits = model(*batch_args, **batch_kwargs)[0]
        result['batch2_logits_max_abs_diff'] = {
            name: float((single_logits[i] - merged_logits[i:i + 1]).abs().max())
            for i, name in enumerate(arms)
        }

        if args_compile_enabled():
            try:
                compiled = torch.compile(model, mode='reduce-overhead')
                result['batch2_compile'] = timed(
                    lambda: compiled(*batch_args, **batch_kwargs), max(reps // 2, 3), device)
            except Exception as exc:  # 编译失败只记录，不中断基准
                result['batch2_compile'] = dict(error=f'{type(exc).__name__}: {exc}')

        try:
            static_args, static_kwargs = _stack_call(batch_args, batch_kwargs, copies=1)
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    model(*static_args, **static_kwargs)
            torch.cuda.current_stream(device).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                model(*static_args, **static_kwargs)
            result['batch2_cuda_graph'] = timed(lambda: graph.replay(), reps, device)
        except Exception as exc:
            result['batch2_cuda_graph'] = dict(error=f'{type(exc).__name__}: {exc}')
    return result


_COMPILE_FLAG = False


def args_compile_enabled():
    return _COMPILE_FLAG


def main():
    global _COMPILE_FLAG
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/eme_long_memory_v2_online_dd_tail_v1.json')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--snr', type=int, default=0)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--reps', type=int, default=20)
    parser.add_argument('--mode', choices=['step', 'forward', 'both'], default='both')
    parser.add_argument('--compile', action='store_true', dest='compile_flag')
    parser.add_argument('--device', default='cuda')
    argparse_args = parser.parse_args()
    _COMPILE_FLAG = argparse_args.compile_flag
    config = json.loads(Path(argparse_args.config).read_text(encoding='utf8'))
    manifest = json.loads(Path(argparse_args.manifest).read_text(encoding='utf8'))
    operator_backend = manifest['variants'][0].get('operator_backend', 'fft')
    os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = operator_backend
    torch.set_num_threads(1)
    device = torch.device(argparse_args.device)
    output = {'seed': argparse_args.seed, 'snr_db': argparse_args.snr,
              'device': argparse_args.device, 'mode': argparse_args.mode,
              'linear_operator_backend': operator_backend}
    if argparse_args.mode in {'step', 'both'}:
        output['step'] = bench_step(config, manifest, device, argparse_args.seed,
                                    argparse_args.snr, argparse_args.frames)
    if argparse_args.mode in {'forward', 'both'}:
        output['forward'] = bench_forward(config, manifest, device, argparse_args.seed,
                                          argparse_args.snr, argparse_args.reps)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
