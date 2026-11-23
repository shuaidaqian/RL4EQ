# -*- coding: utf-8 -*-
"""只读等价性检查：批量/CUDA Graph 接收与逐臂基线在同一波形上的差异与速度。

不写实验结论、不做 BER 确认；用于验证加速实现可安全用于新预登记实验。
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from evaluation.channel_dynamics import load_ar1_dynamics
from evaluation.pilot_transfer import _load_branch
from training.continual_accel import BatchedForward, GraphBatchedForward
from training.continual_receiver import (
    ContinualReceiver, acquisition_transition_intervals, estimate_acquisition, receiver_frame,
)
from training.pilot_transfer import context_model_inputs, frame_to_device


def synchronize(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


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
            dynamics = load_ar1_dynamics(
                resolve(row['dynamics_file']), expected_sha256=row['dynamics_file_sha256'],
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


def numeric_diagnostics(diagnostics):
    return {key: value for key, value in diagnostics.items()
            if isinstance(value, (int, float)) and math.isfinite(value)}


def run_baseline(model, receivers, frames, device, arms):
    synchronize(device)
    samples, logits_by_frame, recs = [], [], []
    for frame in frames:
        began = time.perf_counter()
        frame_logits, frame_records = {}, {}
        for name in arms:
            logits, diagnostics = receivers[name].step(model, frame)
            frame_logits[name] = logits.detach()
            frame_records[name] = numeric_diagnostics(diagnostics)
        synchronize(device)
        samples.append((time.perf_counter() - began) * 1000)
        logits_by_frame.append(frame_logits)
        recs.append(frame_records)
    return samples, logits_by_frame, recs


def run_accelerated(model, decoder, receivers, frames, device, arms):
    synchronize(device)
    samples, logits_by_frame, recs = [], [], []
    for frame in frames:
        began = time.perf_counter()
        prepared = {name: receivers[name].prepare(frame) for name in arms}
        calls = [context_model_inputs(model, prepared[name].context, visible='data') for name in arms]
        with torch.no_grad():
            batch_logits = decoder(calls)
        frame_logits, frame_records = {}, {}
        for index, name in enumerate(arms):
            logits, diagnostics = receivers[name].finish(prepared[name], batch_logits[index])
            frame_logits[name] = logits.detach().clone()
            frame_records[name] = numeric_diagnostics(diagnostics)
        synchronize(device)
        samples.append((time.perf_counter() - began) * 1000)
        logits_by_frame.append(frame_logits)
        recs.append(frame_records)
    return samples, logits_by_frame, recs


def ber_difference(baseline_logits, accelerated_logits, original, device, arms):
    stats = {name: {'error_differences': 0, 'frames_with_differing_errors': 0} for name in arms}
    for frame, base, accel in zip(original, baseline_logits, accelerated_logits):
        mask = frame.data_mask.to(device)
        bits = frame.bits.to(device).bool()
        for name in arms:
            base_errors = int(((base[name][mask] >= 0) != bits[mask]).sum())
            accel_errors = int(((accel[name][mask] >= 0) != bits[mask]).sum())
            stats[name]['error_differences'] += abs(base_errors - accel_errors)
            stats[name]['frames_with_differing_errors'] += int(base_errors != accel_errors)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/eme_long_memory_v2_online_dd_tail_v1.json')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--snr', type=int, default=0)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument('--mode', choices=['batch', 'graph', 'both'], default='both')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf8'))
    backend = manifest['variants'][0].get('operator_backend', 'fft')
    os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = backend
    torch.set_num_threads(1)
    device = torch.device(args.device)
    arms = [row['name'] for row in manifest['variants']]
    if len(arms) < 2:
        raise ValueError('等价性检查至少需要两条臂。')
    env, base_receivers = build_receivers(config, manifest, device, args.seed, args.snr)
    original = [env.next_frame() for _ in range(args.frames)]
    visible = [receiver_frame(frame_to_device(frame, device)) for frame in original]
    model, _ = _load_branch(resolve(manifest['variants'][0]['checkpoint']), config, device)
    model.eval()
    baseline_ms, baseline_logits, baseline_recs = run_baseline(
        model, base_receivers, visible, device, arms)
    report = {
        'seed': args.seed, 'snr_db': args.snr, 'frames': args.frames, 'device': args.device,
        'checkpoint': manifest['variants'][0]['checkpoint'],
        'baseline_frame_mean_ms': statistics.mean(baseline_ms),
        'baseline_frame_median_ms': statistics.median(baseline_ms),
        'modes': {},
    }
    for mode in (['batch', 'graph'] if args.mode == 'both' else [args.mode]):
        _, accel_receivers = build_receivers(config, manifest, device, args.seed, args.snr)
        decoder = (BatchedForward(model) if mode == 'batch' else GraphBatchedForward(model))
        accel_ms, accel_logits, accel_recs = run_accelerated(
            model, decoder, accel_receivers, visible, device, arms)
        steady = accel_ms[1:] if len(accel_ms) > 1 else accel_ms
        entry = {
            'steady_frame_mean_ms': statistics.mean(steady),
            'steady_frame_median_ms': statistics.median(steady),
            'first_frame_ms_including_capture': accel_ms[0],
            'speedup_mean': statistics.mean(baseline_ms) / statistics.mean(steady),
            'speedup_median': statistics.median(baseline_ms) / statistics.median(steady),
            'ber': ber_difference(baseline_logits, accel_logits, original, device, arms),
            'per_arm': {},
        }
        for name in arms:
            entry['per_arm'][name] = {
                'max_logits_abs_diff': max(float((b[name] - a[name]).abs().max())
                                           for b, a in zip(baseline_logits, accel_logits)),
                'diagnostic_max_abs_diff': max(
                    abs(b[name][key] - a[name][key])
                    for b, a in zip(baseline_recs, accel_recs)
                    for key in b[name] if key in a[name] and isinstance(b[name][key], (int, float))),
            }
        report['modes'][mode] = entry
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
