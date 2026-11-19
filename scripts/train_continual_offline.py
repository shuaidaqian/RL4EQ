# -*- coding: utf-8 -*-
"""匹配预算的 acquisition/递归 Pilot 条件离线训练，训练会话不用于确认。"""

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

from agent.unfolded_equalizer import UnfoldedConfig
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from evaluation.seed_registry import reserve_seeds
from training.checkpointing import save_checkpoint
from training.continual_offline import InputRecorder, train_examples
from training.continual_receiver import ContinualReceiver, estimate_acquisition, receiver_frame
from training.pilot_transfer import frame_to_device
from training.rl_modulated_online import _build_equalizer


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/eme_long_memory_v2_online_dd_tail_v1.json')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--training-seeds', nargs='+', type=int, required=True)
    parser.add_argument('--frames', type=int, default=8)
    parser.add_argument('--sample-frames', nargs='+', type=int, default=[0, 1, 3, 7])
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--process-noise', type=float, default=1e-4)
    parser.add_argument('--phase-estimator', choices=['unwrap_median', 'circular_all', 'circular_clean'], default='unwrap_median')
    parser.add_argument('--operator-backend', choices=['direct', 'fft'], default='direct')
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    repeats = len(args.training_seeds)
    if not repeats or len(args.seeds) % repeats:
        parser.error('会话种子必须能按训练重复等分。')
    episodes = len(args.seeds)//repeats
    if episodes < 4 or episodes % 4:
        parser.error('每个训练重复的会话数须为四的倍数，匹配四 SNR 数量。')
    if (args.frames < 1 or not args.sample_frames or len(set(args.sample_frames)) != len(args.sample_frames)
            or min(args.sample_frames) < 0 or max(args.sample_frames) >= args.frames):
        parser.error('训练帧选择必须非空、互异且位于接收会话内。')
    if not 0 < args.learning_rate < 1:
        parser.error('学习率须为零到一之间的有限数。')
    output = Path(args.output_dir)
    report = Path(args.report)
    if output.exists() or report.exists():
        raise ValueError('权重目录或报告已存在，禁止隐式重跑或覆盖。')
    torch.set_num_threads(1)
    os.environ['RL4EQ_LINEAR_OPERATOR_BACKEND'] = args.operator_backend
    device = torch.device(args.device)
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    model_config = UnfoldedConfig.from_dict(config['model'])
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if model_config.to_dict() != UnfoldedConfig.from_dict(payload['model_config']).to_dict():
        raise ValueError('禁止静默更改起始 checkpoint 模型配置。')
    protocol = dict(vars(args), role='training', repeats=repeats, episodes_per_repeat=episodes,
                    methods=['acquisition', 'recursive_pilot'], snr_schedule=[0, 5, 10, 15]*(episodes//4),
                    supervision='仅离线 Data BCE，接收输入不含 Data/Reward 标签',
                    trajectory='每个会话内权重固定，轨迹结束后训练；历史状态截断梯度',
                    inference_mode='eval，保留梯度；在线部署权重冻结',
                    config_sha256=fingerprint(args.config), checkpoint_sha256=fingerprint(args.checkpoint),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    sources={name:fingerprint(ROOT/name) for name in [
                        'training/continual_receiver.py', 'training/continual_offline.py',
                        'scripts/train_continual_offline.py', 'agent/unfolded_equalizer.py']})
    reserve_seeds(ROOT, experiment=args.experiment, seeds=args.seeds+args.training_seeds,
                  purpose='training', details=protocol)
    output.mkdir(parents=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    result = dict(protocol=protocol, rows=[], checkpoints=[], status='running')

    def save():
        target = report
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        temporary.replace(target)

    save()
    try:
        for repeat, training_seed in enumerate(args.training_seeds):
            models, recorders, optimizers = {}, {}, {}
            for arm in protocol['methods']:
                # 两方法同起点、同优化器和全参数集合，不挂载在线适配器。
                torch.manual_seed(training_seed)
                model = _build_equalizer(model_config, Path(args.checkpoint), str(device))
                for parameter in model.parameters():
                    parameter.requires_grad_(True)
                models[arm] = model
                recorders[arm] = InputRecorder(model)
                optimizers[arm] = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
            order_generator = torch.Generator().manual_seed(training_seed)
            for episode in range(episodes):
                seed = args.seeds[repeat*episodes+episode]
                snr = protocol['snr_schedule'][episode]
                env = CommunicationEnvironment(build_comm_env_config(config, level='B', snr_db=snr, seed=seed,
                    max_delay=config['max_delay'], total_pilot=config['pilot_total'],
                    reward_pilot_total=config['reward_pilot_total'], pilot_layout='prefix', state_split='offline_train'))
                start = env.reset_episode()
                acquisition = estimate_acquisition(start.acquisition.tx_symbols, start.acquisition.rx_symbols,
                    config['max_delay'], cfo_limit=config['acquisition_cfo_limit'])
                acquisition = replace(acquisition, mean=acquisition.mean.to(device),
                                      covariance=acquisition.covariance.to(device))
                receivers = {arm:ContinualReceiver(acquisition, start.initial_soft_tail.to(device),
                    arm=arm, process_noise=args.process_noise, phase_estimator=args.phase_estimator) for arm in models}
                examples = {arm:[] for arm in models}
                began = time.perf_counter()
                # 同一波形只生成一次，两模型独立生成自己的 soft-tail 和接收状态。
                for frame_index in range(args.frames):
                    original = frame_to_device(env.next_frame(), device)
                    visible = receiver_frame(original)
                    for arm in models:
                        receivers[arm].step(recorders[arm], visible)
                        if frame_index in args.sample_frames:
                            examples[arm].append(recorders[arm].example(
                                original.bits, original.data_mask, frame_index))
                order = torch.randperm(len(args.sample_frames), generator=order_generator).tolist()
                for arm in models:
                    metrics = train_examples(models[arm], optimizers[arm], examples[arm], order=order)
                    result['rows'].append(dict(repeat=repeat, training_seed=training_seed,
                        episode=episode, seed=seed, snr_db=snr, method=arm, metrics=metrics,
                        rollout_forwards=args.frames, gradient_forwards=len(metrics)))
                    save_checkpoint(output/f'repeat{repeat}_{arm}.pt', dict(
                        schema_version='unfolded-eq-v1', model_config=model_config.to_dict(),
                        state_dict=models[arm].state_dict(), optimizer=optimizers[arm].state_dict(),
                        training_protocol=protocol, repeat=repeat, completed_episodes=episode+1))
                save()
                print(f'离线训练 repeat={repeat} episode={episode+1}/{episodes} SNR={snr} '
                      f'耗时={time.perf_counter()-began:.1f}s '+ ' '.join(
                          f"{arm}={sum(m['data_bce'] for m in result['rows'][-2+i]['metrics'])/len(order):.5f}"
                          for i, arm in enumerate(models)), flush=True)
            for arm in models:
                checkpoint = output/f'repeat{repeat}_{arm}.pt'
                result['checkpoints'].append(dict(repeat=repeat, method=arm,
                    path=str(checkpoint), sha256=fingerprint(checkpoint)))
            del models, recorders, optimizers, receivers, examples
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        result['status'] = 'completed'
    except BaseException as exc:
        result['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        result['failure'] = dict(type=type(exc).__name__, message=str(exc))
        save()
        raise
    save()


if __name__ == '__main__':
    main()
