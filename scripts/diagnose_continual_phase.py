# -*- coding: utf-8 -*-
"""新会话首帧同步诊断：真发送符号只在估计结束后用于预测评分。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config
from env.linear_operator import LinearChannelOperator
from evaluation.continual_summary import interval
from evaluation.seed_registry import reserve_seeds
from training.continual_phase import circular_phase_features
from training.continual_receiver import estimate_acquisition, phase_features, receiver_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/eme_long_memory_v2_online_dd_tail_v1.json')
    parser.add_argument('--seeds', nargs='+', type=int, required=True)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('禁止覆盖已登记实验。')
    torch.set_num_threads(1)
    config = json.loads(Path(args.config).read_text(encoding='utf8'))
    protocol = dict(vars(args), role='development', frames=1, snrs=[0, 5, 10, 15],
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    scope='仅首帧同步与预测误差，不证明神经 BER 或递归增长',
                    phase_source_sha256=hashlib.sha256((ROOT/'training/continual_phase.py').read_bytes()).hexdigest())
    reserve_seeds(ROOT, experiment=args.experiment, seeds=args.seeds, purpose='development', details=protocol)
    result = dict(protocol=protocol, rows=[], status='running')
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
        temporary.replace(output)

    save()
    for snr in protocol['snrs']:
        for seed in args.seeds:
            env = CommunicationEnvironment(build_comm_env_config(config, level='B', snr_db=snr, seed=seed,
                max_delay=config['max_delay'], total_pilot=config['pilot_total'],
                reward_pilot_total=config['reward_pilot_total'], pilot_layout='prefix', state_split='offline_train'))
            start = env.reset_episode()
            acq = estimate_acquisition(start.acquisition.tx_symbols, start.acquisition.rx_symbols,
                                       config['max_delay'], cfo_limit=config['acquisition_cfo_limit'])
            original = env.next_frame()
            visible = receiver_frame(original)
            features = {
                'unwrap_median': phase_features(visible, acq.mean, start.initial_soft_tail, acq.cfo),
                'circular_all': circular_phase_features(visible, acq.mean, start.initial_soft_tail, acq.cfo),
                'circular_clean': circular_phase_features(visible, acq.mean, start.initial_soft_tail, acq.cfo, clean_rows=True),
            }
            # 到这里才读普通帧真实符号；不进入上述相位估计或 acquisition。
            operator = LinearChannelOperator(frame_len=config['frame_len'], max_delay=config['max_delay'])
            prediction = operator.forward(original.tx_symbols[None], acq.mean[None], start.initial_soft_tail[None]).flatten()
            positions = torch.arange(config['frame_len'])
            row = dict(seed=seed, snr_db=snr, noise_estimate=acq.noise_variance, methods={})
            for name, vector in features.items():
                fitted = prediction * torch.exp(1j*(vector[0]+2*torch.pi*vector[1]*positions))
                errors = (original.rx_symbols-fitted).abs().square()
                row['methods'][name] = dict(phase=float(vector[0]),
                    reward_prediction_mse=float(errors[original.reward_mask].mean()),
                    data_prediction_mse=float(errors[original.data_mask].mean()))
            result['rows'].append(row)
        save()
        print(f'同步诊断 SNR={snr} 完成，{len(args.seeds)} 个独立首帧。', flush=True)
    result['summary'] = {}
    for snr in protocol['snrs']:
        rows = [r for r in result['rows'] if r['snr_db'] == snr]
        result['summary'][str(snr)] = {
            method: {metric: interval([r['methods'][method][metric] for r in rows])
                     for metric in ['reward_prediction_mse', 'data_prediction_mse']}
            for method in features
        }
        for method in ['circular_all', 'circular_clean']:
            result['summary'][str(snr)][method]['paired_data_mse_reduction_vs_unwrap'] = interval([
                r['methods']['unwrap_median']['data_prediction_mse']-r['methods'][method]['data_prediction_mse']
                for r in rows])
    result['status'] = 'completed'
    save()


if __name__ == '__main__':
    main()
