# -*- coding: utf-8 -*-
"""用预先声明的候选清单作同波形开发比较，不覆盖旧实验。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
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
from evaluation.pilot_transfer import _load_branch
from evaluation.receiver_diagnostics import receiver_diagnostics
from evaluation.seed_registry import reserve_seeds
from scripts.develop_continual_receiver import fingerprint, summarize, synchronize
from training.continual_receiver import ContinualReceiver, estimate_acquisition, receiver_frame
from training.continual_offline import InputRecorder
from training.pilot_transfer import frame_to_device


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
    parser.add_argument('--checkpoint-interval', type=int, default=8,
                        help='按多少帧原子保存一次；流式长会话可设为较大值。')
    args = parser.parse_args()
    if args.frames < 1 or args.checkpoint_interval < 1 or not args.snrs or len(set(args.snrs)) != len(args.snrs):
        parser.error('需要正帧数和互异的 SNR。')
    output = Path(args.output)
    rows_output = output.with_suffix('.jsonl')
    if output.exists() or (args.stream_rows and rows_output.exists()):
        raise ValueError('禁止覆盖或隐式重跑旧实验。')
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf8'))
    variants = manifest.get('variants', [])
    for row in variants:
        if row.get('state_model', 'random_walk') == 'ar1' and row.get('dynamics_file'):
            dynamics_path = Path(row['dynamics_file'])
            if not dynamics_path.is_absolute():
                dynamics_path = ROOT / dynamics_path
            expected_hash = row.get('dynamics_file_sha256')
            if not expected_hash or fingerprint(dynamics_path) != expected_hash:
                raise ValueError(f"AR(1) 离线统计文件指纹不匹配：{dynamics_path}")
            dynamics = json.loads(dynamics_path.read_text(encoding='utf8'))
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
            branch = KnownAdaptRealPhysicsBranch(model)
        else:
            branch = PhysicsBranch(model)
        recorders[row['name']] = InputRecorder(branch)
        models[row['name']] = ForwardCounter(recorders[row['name']])
    protocol = dict(vars(args), variants=variants, arms=[row['name'] for row in variants],
                    pairs=manifest.get('pairs', []), role='development', noise_source='acquisition_residual',
                    statistical_unit='session', timing='current_pilot_then_decode',
                    linear_operator_backend=os.environ.get('RL4EQ_LINEAR_OPERATOR_BACKEND', 'direct'),
                    torch_version=torch.__version__, config_sha256=fingerprint(args.config),
                    manifest_sha256=fingerprint(args.manifest),
                    checkpoint_sha256={p:fingerprint(p) for p in cache},
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    sources={name:fingerprint(ROOT/name) for name in [
                        'training/continual_receiver.py', 'training/continual_phase.py',
                        'evaluation/continual_variants.py', 'scripts/compare_continual_variants.py',
                        'evaluation/receiver_diagnostics.py',
                        'agent/unfolded_equalizer.py', 'env/linear_operator.py']})
    reserve_seeds(ROOT, experiment=args.experiment, seeds=args.seeds, purpose='development', details=protocol)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(protocol=protocol, acquisitions=[], rows=[], status='running')
    streamed_rows = 0
    session_errors, session_bits = {}, {}

    def save():
        nonlocal streamed_rows
        if args.stream_rows:
            if result['rows']:
                with rows_output.open('a', encoding='utf8', newline='\n') as stream:
                    for row in result['rows']:
                        compact = {key:row[key] for key in (
                            'seed', 'snr_db', 'frame_index', 'arm', 'errors', 'bits', 'elapsed_ms', 'rx_sha256',
                        )}
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
        temporary.replace(output)

    save()
    try:
        for snr in args.snrs:
            for seed in args.seeds:
                env = CommunicationEnvironment(build_comm_env_config(config, level='B', snr_db=snr, seed=seed,
                    max_delay=config['max_delay'], total_pilot=config['pilot_total'],
                    reward_pilot_total=config['reward_pilot_total'], pilot_layout='prefix', state_split=args.state_split))
                start = env.reset_episode()
                acquisition = estimate_acquisition(start.acquisition.tx_symbols, start.acquisition.rx_symbols,
                    config['max_delay'], cfo_limit=config['acquisition_cfo_limit'])
                result['acquisitions'].append(dict(seed=seed, snr_db=snr,
                    noise_estimate=acquisition.noise_variance, noise_truth_diagnostic=10**(-snr/10),
                    cfo_estimate=acquisition.cfo, cir_norm=float(acquisition.mean.norm())))
                acquisition = replace(acquisition, mean=acquisition.mean.to(device),
                                      covariance=acquisition.covariance.to(device))
                receivers = {row['name']:ContinualReceiver(acquisition, start.initial_soft_tail.to(device),
                    arm=row['arm'], process_noise=row.get('process_noise', 1e-4),
                    phase_estimator=row.get('phase_estimator', 'unwrap_median'),
                    phase_reference_backend=row.get('phase_reference_backend', 'vectorized'),
                    state_model=row.get('state_model', 'random_walk'),
                    state_rho=row.get('state_rho'),
                    stationary_power=row.get('stationary_power_by_tap')) for row in variants}
                for _ in range(args.frames):
                    original = frame_to_device(env.next_frame(), device)
                    visible = receiver_frame(original)
                    rx_hash = hashlib.sha256(original.rx_symbols.cpu().numpy().tobytes()).hexdigest()
                    frame_logits, frame_diagnostics = {}, {}
                    for row in variants:
                        name = row['name']
                        decoder = models[name]
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
                                logits, diagnostics = receivers[name].step(decoder, visible)
                        finally:
                            if previous_backend is None:
                                os.environ.pop('RL4EQ_LINEAR_OPERATOR_BACKEND', None)
                            else:
                                os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = previous_backend
                        synchronize(device)
                        elapsed = (time.perf_counter()-began)*1000
                        frame_diagnostics[name] = dict(diagnostics)
                        # 接收已结束，以下全部为事后评分，不回流任何更新。
                        mask = original.data_mask
                        errors = int(((logits[mask]>=0) != original.bits[mask].bool()).sum())
                        session_key = (snr, seed, name)
                        session_errors[session_key] = session_errors.get(session_key, 0) + errors
                        session_bits[session_key] = session_bits.get(session_key, 0) + int(mask.sum())
                        estimate = diagnostics.pop('decode_cir')
                        truth = original.true_cir.flatten()
                        scoring = receiver_diagnostics(logits, original.bits, mask, estimate, truth)
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
                            elapsed_ms=elapsed, model_calls=decoder.calls-before, **scoring,
                            paired_logits_diagnostic=paired_logits,
                            noise_variance_used=acquisition.noise_variance, rx_sha256=rx_hash, **diagnostics))
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
