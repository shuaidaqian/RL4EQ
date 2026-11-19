# -*- coding: utf-8 -*-
"""闭环训练交互的交叉统计：区分训练重复与接收会话两类随机性。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from evaluation.continual_summary import interval, summarize_development


def summarize_training_interaction(result):
    """不把同一会话下的三个训练模型当成三个独立接收会话。"""
    summarize_development(result)
    protocol = result['protocol']
    variants = protocol.get('variants', [])
    mapping = {}
    for row in variants:
        if 'training_repeat' not in row:
            continue
        key = (row['training_repeat'], row['training_method'], row['arm'])
        if key in mapping:
            raise ValueError('同一训练重复/策略/接收臂只能对应一个模型。')
        mapping[key] = row['name']
    repeats = sorted({key[0] for key in mapping})
    if len(repeats) < 2:
        raise ValueError('至少保留两个训练重复，不能只报告挑选的单模型。')
    for repeat in repeats:
        for method in ['acquisition', 'recursive_pilot']:
            for arm in ['current_pilot', 'recursive_pilot']:
                if (repeat, method, arm) not in mapping:
                    raise ValueError('训练策略与接收臂的交叉设计不完整。')
    scores, waveforms = defaultdict(lambda:[0, 0]), {}
    for row in result['rows']:
        identity = (row['snr_db'], row['seed'], row['frame_index'])
        if identity in waveforms and waveforms[identity] != row['rx_sha256']:
            raise ValueError('训练模型之间接收波形不匹配。')
        waveforms[identity] = row['rx_sha256']
        score = scores[(row['snr_db'], row['seed'], row['arm'])]
        score[0] += row['errors']
        score[1] += row['bits']

    def ber(snr, seed, repeat, method, arm):
        errors, bits = scores[(snr, seed, mapping[(repeat, method, arm)])]
        return errors/bits

    summary = dict(status='development_descriptive_only', per_snr={},
        interpretation='交互为(闭环训练B-C)-(参照训练B-C)，正值表示闭环训练增强递归增量；没有总体随机性联合置信区间。',
        repeats=repeats, receiver_seeds=protocol['seeds'],
        limitation='报告分别条件于已训练模型和已评价会话的两个统计轴；不能把交叉单元当独立样本，也不能以小预算阴性结果排除充分重训的价值。')
    for snr in protocol['snrs']:
        interactions = []
        details = {}
        for repeat in repeats:
            reference, closed, values = [], [], []
            for seed in protocol['seeds']:
                first = ber(snr, seed, repeat, 'acquisition', 'current_pilot')-ber(snr, seed, repeat, 'acquisition', 'recursive_pilot')
                second = ber(snr, seed, repeat, 'recursive_pilot', 'current_pilot')-ber(snr, seed, repeat, 'recursive_pilot', 'recursive_pilot')
                reference.append(first)
                closed.append(second)
                values.append(second-first)
            interactions.append(values)
            details[str(repeat)] = dict(reference_recursive_gain=interval(reference),
                closed_recursive_gain=interval(closed), interaction=interval(values),
                interaction_by_receiver_seed=dict(zip(map(str, protocol['seeds']), values)))
        matrix = np.asarray(interactions)
        summary['per_snr'][str(snr)] = dict(per_repeat=details, mean_interaction=float(matrix.mean()),
            by_receiver_session_conditioning_on_trained_models=interval(matrix.mean(0)),
            by_training_repeat_conditioning_on_receiver_sessions=interval(matrix.mean(1)),
            joint_generalization_ci=None)
    return summary


def interaction_report(summary):
    lines = ['# 离线训练与在线递归的交互开发分析', '',
             '**只作开发诊断，未建立四 SNR 增长至饱和或训练策略的总体优势。**', '',
             summary['interpretation'], '', summary['limitation'], '',
             '| SNR | 训练重复 | 参照训练 B-C | 闭环训练 B-C | 交互 | 会话轴 95% 区间 |',
             '|---|---|---:|---:|---:|---|']
    for snr, value in summary['per_snr'].items():
        for repeat, row in value['per_repeat'].items():
            stat = row['interaction']
            ci = '样本不足或方差退化' if stat['ci95'] is None else f"[{stat['ci95'][0]:+.6f}, {stat['ci95'][1]:+.6f}]"
            lines.append(f"| {snr} | {repeat} | {row['reference_recursive_gain']['mean']:+.6f} | "
                         f"{row['closed_recursive_gain']['mean']:+.6f} | {stat['mean']:+.6f} | {ci} |")
    lines += ['', '三个模型在同一接收会话上的结果相关。JSON 分别给出会话轴、训练重复轴统计，未将二者合并成独立样本。', '']
    return '\n'.join(lines)
