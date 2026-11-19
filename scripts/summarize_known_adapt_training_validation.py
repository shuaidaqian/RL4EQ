# -*- coding: utf-8 -*-
"""汇总新物理 warm-start 微调验证，分开接收会话与训练重复统计。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    margin = float(t.ppf(.975, len(values)-1) * sd / np.sqrt(len(values))) if len(values)>1 and sd>0 else None
    return {"mean":mean,"ci95":None if margin is None else [mean-margin,mean+margin],"n":len(values),
            "status":"zero_empirical_variance_not_equivalence" if len(values)>1 and sd==0 else "t_descriptive"}


def summarize(result):
    if result.get('status') != 'completed':
        raise ValueError('只汇总完整实验。')
    protocol=result['protocol']; rows=result['rows']; expected=int(protocol['frames'])
    arms=protocol['arms']; by=defaultdict(dict)
    for row in rows:
        key=(row['snr_db'],row['seed'],row['arm'])
        if row['frame_index'] in by[key]: raise ValueError('重复帧。')
        by[key][row['frame_index']]=row
    for key,frames in by.items():
        if set(frames)!=set(range(expected)): raise ValueError(f'会话缺帧：{key}')
    variants={v['name']:v for v in protocol['variants']}
    repeats=sorted({v['training_repeat'] for v in protocol['variants'] if 'training_repeat' in v})
    seeds=protocol['seeds']
    def ber(snr,seed,arm):
        fs=by[(snr,seed,arm)]
        return sum(x['errors'] for x in fs.values())/sum(x['bits'] for x in fs.values())
    out={"status":"development_descriptive_only","statistical_axes":{
        "receiver_session":"10 paired receiver sessions, conditional on each trained model",
        "training_repeat":"3 independent training repeats; each cell averages the same 10 receiver sessions",
        "joint_generalization_ci":None},"per_snr":{}}
    for snr in protocol['snrs']:
        values={arm:[ber(snr,seed,arm) for seed in seeds] for arm in arms}
        snrout={"mean_ber_by_candidate":{arm:interval(v) for arm,v in values.items()},
                "untrained_neural_minus_trained_by_repeat":{},"recursive_minus_acquisition_training_by_repeat":{}}
        for repeat in repeats:
            candidates={method:f'train_r{repeat}_{method}_known_adapt_neural' for method in ('acquisition','recursive_pilot')}
            snrout['untrained_neural_minus_trained_by_repeat'][str(repeat)]={
                method:interval([ber(snr,seed,'untrained_known_adapt_neural')-ber(snr,seed,name) for seed in seeds])
                for method,name in candidates.items()}
            # 训练策略的统计单位是训练重复：先在接收会话内平均，再以 3 个重复作 t 区间。
            snrout['recursive_minus_acquisition_training_by_repeat'][str(repeat)]=float(np.mean([
                ber(snr,seed,candidates['acquisition'])-ber(snr,seed,candidates['recursive_pilot'])
                for seed in seeds]))
        snrout['recursive_minus_acquisition_training_repeat_axis']=interval(list(
            snrout['recursive_minus_acquisition_training_by_repeat'].values()))
        snrout['pilot_physics_gain_vs_acquisition']=interval([ber(snr,seed,'acquisition_physics_q1e6')-
                                                               ber(snr,seed,'recursive_pilot_physics_q1e6') for seed in seeds])
        snrout['known_adapt_physics_gain_vs_recursive_pilot']=interval([ber(snr,seed,'recursive_pilot_physics_q1e6')-
                                                                         ber(snr,seed,'untrained_known_adapt_physics') for seed in seeds])
        out['per_snr'][str(snr)]=snrout
    return out


def report(summary):
    lines=['# 新 warm-start 离线训练验证','',
      '**开发验证；不证明增长至饱和。会话轴条件于已训练模型，训练重复轴以每重复的会话均值为单位。**','',
      '| SNR | Frozen acquisition BER | 递归 Pilot 物理 BER | 已知 Adapt 实值 BER |',
      '|---:|---:|---:|---:|']
    for snr,v in summary['per_snr'].items():
        a=v['mean_ber_by_candidate']; lines.append(f"| {snr} | {a['acquisition_physics_q1e6']['mean']:.6f} | {a['recursive_pilot_physics_q1e6']['mean']:.6f} | {a['untrained_known_adapt_physics']['mean']:.6f} |")
    lines+=['','| SNR | 训练重复 | acquisition 训练相对未训练 BER 改善 | recursive Pilot 训练相对未训练 BER 改善 | recursive−acquisition 策略差 |',
            '|---:|---:|---:|---:|---:|']
    for snr,v in summary['per_snr'].items():
        for rep,row in v['untrained_neural_minus_trained_by_repeat'].items():
            aa=row['acquisition']['mean']; rr=row['recursive_pilot']['mean']; diff=v['recursive_minus_acquisition_training_by_repeat'][rep]
            lines.append(f'| {snr} | {rep} | {aa:+.7f} | {rr:+.7f} | {diff:+.7f} |')
    lines+=['','训练策略差的区间以 3 个独立训练重复为单位；交叉会话不能伪作额外训练重复。未建立联合泛化区间。','']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--input',required=True); args=parser.parse_args()
    source=Path(args.input); result=json.loads(source.read_text(encoding='utf8')); summary=summarize(result)
    summary['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source.with_name(source.stem+'_training_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    source.with_name(source.stem+'_training_report.md').write_text(report(summary),encoding='utf8')
    print(report(summary))

if __name__=='__main__': main()
