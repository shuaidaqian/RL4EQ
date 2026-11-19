# -*- coding: utf-8 -*-
"""分开统计 64 与 256 次更新模型的接收会话轴和训练重复轴。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import t

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def ci(values):
    x=np.asarray(values,dtype=float); m=float(x.mean()); sd=float(x.std(ddof=1)) if len(x)>1 else 0.0
    margin=float(t.ppf(.975,len(x)-1)*sd/np.sqrt(len(x))) if len(x)>1 and sd>0 else None
    return {'mean':m,'ci95':None if margin is None else [m-margin,m+margin],
            'n':len(x),'status':'zero_empirical_variance_not_equivalence' if len(x)>1 and sd==0 else 't_descriptive'}


def summarize(result):
    if result.get('status')!='completed': raise ValueError('仅分析完整实验。')
    p=result['protocol']; frame_count=int(p['frames']); seeds=p['seeds']; snrs=p['snrs']; rows=result['rows']
    grouped=defaultdict(dict)
    for row in rows:
        k=(row['snr_db'],row['seed'],row['arm'])
        if row['frame_index'] in grouped[k]: raise ValueError('重复帧。')
        grouped[k][row['frame_index']]=row
    for k,v in grouped.items():
        if set(v)!=set(range(frame_count)): raise ValueError(f'会话缺帧：{k}')
    def frame_ber(snr,seed,arm,lo,hi):
        rs=[grouped[(snr,seed,arm)][i] for i in range(lo,hi)]
        return sum(x['errors'] for x in rs)/sum(x['bits'] for x in rs)
    out={'status':'development_descriptive_only','receiver_session_unit':'独立波形会话；模型比较逐会话配对',
         'training_repeat_unit':'先平均接收会话，再以三个训练重复为统计样本','joint_generalization_ci':None,'per_snr':{}}
    for snr in snrs:
        names=['acquisition_physics_q1e6','recursive_pilot_physics_q1e6','untrained_known_adapt_physics','untrained_known_adapt_neural']
        names += [f'train{budget}_r{rep}_recursive_pilot' for budget in (64,256) for rep in range(3)]
        values={name:[frame_ber(snr,seed,name,0,frame_count) for seed in seeds] for name in names}
        v={'mean_ber':{name:ci(vals) for name,vals in values.items()},'own_early_minus_late':{},
           'trained_minus_untrained_by_repeat':{},'train256_minus_train64_by_repeat':{}}
        for name in names:
            v['own_early_minus_late'][name]=ci([frame_ber(snr,s,name,0,4)-frame_ber(snr,s,name,48,64) for s in seeds])
        for budget in (64,256):
            for rep in range(3):
                name=f'train{budget}_r{rep}_recursive_pilot'
                v['trained_minus_untrained_by_repeat'][name]=ci([values['untrained_known_adapt_neural'][i]-values[name][i] for i in range(len(seeds))])
        repeat_diffs=[]
        for rep in range(3):
            a=f'train64_r{rep}_recursive_pilot'; b=f'train256_r{rep}_recursive_pilot'
            diff=ci([values[a][i]-values[b][i] for i in range(len(seeds))])
            v['train256_minus_train64_by_repeat'][str(rep)]=diff
            repeat_diffs.append(values[a] and float(np.mean([values[a][i]-values[b][i] for i in range(len(seeds))])))
        v['training_repeat_axis_train256_minus_train64']=ci(repeat_diffs)
        v['physics_gain_vs_acquisition']=ci([values['acquisition_physics_q1e6'][i]-values['untrained_known_adapt_physics'][i] for i in range(len(seeds))])
        v['acquisition_gain_late_minus_early']=ci([
            (frame_ber(snr,seed,'acquisition_physics_q1e6',48,64)-frame_ber(snr,seed,'untrained_known_adapt_physics',48,64))
            -(frame_ber(snr,seed,'acquisition_physics_q1e6',0,4)-frame_ber(snr,seed,'untrained_known_adapt_physics',0,4))
            for seed in seeds
        ])
        out['per_snr'][str(snr)]=v
    return out


def report(s):
    lines=['# 已知 Adapt warm-start 的离线训练预算验证','',
      '**开发筛查，不是独立确认，也不证明增长至饱和。会话轴与训练重复轴分别报告。**','',
      '| SNR | acquisition BER | Pilot 递归 BER | 未训练约束 BER |']
    lines+=['|---:|---:|---:|---:|']
    for snr,v in s['per_snr'].items():
        m=v['mean_ber']; lines.append(f"| {snr} | {m['acquisition_physics_q1e6']['mean']:.6f} | {m['recursive_pilot_physics_q1e6']['mean']:.6f} | {m['untrained_known_adapt_physics']['mean']:.6f} |")
    lines+=['','| SNR | 重复 | 64 更新相对未训练 BER 改善 | 256 更新相对未训练 BER 改善 | 256 相对 64 更新增益 |',
            '|---:|---:|---:|---:|---:|']
    for snr,v in s['per_snr'].items():
        for rep in range(3):
            a=v['trained_minus_untrained_by_repeat'][f'train64_r{rep}_recursive_pilot']['mean']
            b=v['trained_minus_untrained_by_repeat'][f'train256_r{rep}_recursive_pilot']['mean']
            d=v['train256_minus_train64_by_repeat'][str(rep)]['mean']
            lines.append(f'| {snr} | {rep} | {a:+.7f} | {b:+.7f} | {d:+.7f} |')
    lines+=['','正值表示待比较的训练模型 BER 更低。训练重复轴区间基于三个重复各自对十个接收会话的平均差；会话轴区间不把不同 checkpoint 当独立接收会话。','']
    return '\n'.join(lines)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--input',required=True);args=ap.parse_args()
    source=Path(args.input);data=json.loads(source.read_text(encoding='utf-8'));result=summarize(data)
    result['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source.with_name(source.stem+'_budget_summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    source.with_name(source.stem+'_budget_report.md').write_text(report(result),encoding='utf-8')
    print(report(result))

if __name__=='__main__':main()
