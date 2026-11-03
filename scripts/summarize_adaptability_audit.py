# -*- coding: utf-8 -*-
"""核验完整开发审计并归档；不按观察到的收益选择会话或时间段。"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.pilot_transfer import paired_interval


def digest(content):
    return hashlib.sha256(content).hexdigest()


def validate_records(spec, summary, rows, probes):
    """要求完整固定矩阵，探针与主轨迹同波形，数值和参数边界合法。"""
    if summary.get("smoke") or summary.get("core_confirmed") or summary.get("role") != "development":
        raise ValueError("仅归档未冒充正式确认的完整开发运行。")
    if any(summary.get(key) != spec.get(field) for key, field in
           [('seeds','seeds'),('snrs_db','snrs_db'),('frames_per_seed_snr','frames')]):
        raise ValueError("运行汇总与清单的种子、SNR或帧预算不一致。")
    expected = {(seed, snr, index) for seed in spec['seeds'] for snr in spec['snrs_db']
                for index in range(spec['frames'])}
    actual = [(row['seed'], row['snr_db'], row['frame']) for row in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("主轨迹不完整、含额外条件或存在重复帧。")
    expected_probes = {(seed, snr, offset) for seed in spec['seeds'] for snr in spec['snrs_db']
                       for offset in spec['fixed_candidate_offsets']}
    actual_probes = [(row['seed'], row['snr_db'], row['offset']) for row in probes]
    if len(actual_probes) != len(set(actual_probes)) or set(actual_probes) != expected_probes:
        raise ValueError("一次候选探针不完整或存在重复。")
    by_key = dict(zip(actual, rows))
    for row in rows:
        for key in ('ber_F', 'ber_P', 'ber_N', 'ber_T', 'ber_N_candidate'):
            if not np.isfinite(row[key]) or not 0 <= row[key] <= 1:
                raise ValueError("BER 非有限或越界。")
        if row['retained_total_norm'] > spec['deterministic_update']['total_radius'] + 1e-5:
            raise ValueError("神经记忆累计参数越界。")
        if row.get('uses_rl') or row.get('data_labels_used_for_control'):
            raise ValueError("控制信息边界不符合确定性协议。")
        if not row.get('shared_physical_context_for_P_N'):
            raise ValueError("P/N 未共享物理上下文。")
    for row in probes:
        if row.get('future_metrics_used_for_control') or row['source_frame'] != 0:
            raise ValueError("一次候选探针的信息时序无效。")
        for field in ('ber_P','ber_raw','ber_guarded'):
            if not np.isfinite(row[field]) or not 0 <= row[field] <= 1:
                raise ValueError("一次候选探针BER非有限或越界。")
        key = (row['seed'], row['snr_db'], row['offset'])
        if key in by_key:
            for field in ('rx_sha256', 'lookahead_rx_sha256'):
                if row[field] != by_key[key][field]:
                    raise ValueError("一次候选与持续分支波形不一致。")
    return True


def analyze(spec, rows, probes):
    """固定前/中/后窗口只作描述，所有逐seed值均保留。"""
    records = []
    for snr in spec['snrs_db']:
        selected = [row for row in rows if row['snr_db'] == snr]
        intervals = {}
        for name, left, right in [('state_gain_F_minus_P', 'ber_F', 'ber_P'),
                                  ('neural_gain_P_minus_N', 'ber_P', 'ber_N'),
                                  ('diagnostic_gap_P_minus_T', 'ber_P', 'ber_T')]:
            values = [float(np.mean([row[left]-row[right] for row in selected if row['seed']==seed]))
                      for seed in spec['seeds']]
            intervals[name] = dict(**paired_interval(values), per_seed=values)
        segments = {}
        for label, begin, end in [('first_32',0,32), ('middle_64',32,96), ('last_32',96,128)]:
            active = [row for row in selected if begin <= row['frame'] < end]
            segments[label] = {key: float(np.mean([row[key] for row in active]))
                              for key in ('ber_F','ber_P','ber_N','ber_T')}
        fixed = []
        for offset in spec['fixed_candidate_offsets']:
            active = [row for row in probes if row['snr_db']==snr and row['offset']==offset]
            fixed.append(dict(offset=offset, **{key:float(np.mean([row[key] for row in active]))
                         for key in ('ber_P','ber_raw','ber_guarded')}))
        accepted = [row for row in selected if row['accepted']]
        records.append(dict(snr_db=snr, paired_intervals=intervals, descriptive_segments=segments,
            fixed_candidate=fixed, accepted_frames=len(accepted), frames=len(selected),
            accepted_frames_N_worse_than_P=sum(row['ber_N']>row['ber_P'] for row in accepted),
            max_retained_total_norm=max(row['retained_total_norm'] for row in selected),
            timing_ms={key:dict(mean=float(np.mean([row[key] for row in selected])),
                               p95=float(np.percentile([row[key] for row in selected],95)))
                       for key in ('F_ms','P_ms','N_ms','T_ms')},
            geometry=[dict(seed=row['seed'], frame=row['frame'], information=row['information'])
                      for row in selected if row['frame'] in (0,7,127)]))
    return records


def archive(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError("归档目录已存在，禁止覆盖。")
    spec = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    summary = json.loads((source/'summary.json').read_text(encoding='utf-8'))
    raw, probe_raw = (source/'frame_metrics.jsonl').read_bytes(), (source/'fixed_candidate_metrics.jsonl').read_bytes()
    if digest(raw) != summary['raw_sha256'] or digest(probe_raw) != summary['fixed_candidate_raw_sha256']:
        raise ValueError("原始轨迹 SHA256 不一致。")
    rows, probes = [json.loads(line) for line in raw.splitlines()], [json.loads(line) for line in probe_raw.splitlines()]
    validate_records(spec, summary, rows, probes)
    if len(rows)!=summary['records'] or len(probes)!=summary['fixed_candidate_records']:
        raise ValueError("汇总记录数与原始日志不一致。")
    destination.mkdir(parents=True)
    for name in ('manifest.json','summary.json'):
        (destination/name).write_bytes((source/name).read_bytes())
    compressed = {}
    for name, content in [('frame_metrics.jsonl',raw), ('fixed_candidate_metrics.jsonl',probe_raw)]:
        data = gzip.compress(content, mtime=0)
        (destination/(name+'.gz')).write_bytes(data)
        compressed[name] = dict(raw_sha256=digest(content), gzip_sha256=digest(data), raw_bytes=len(content), gzip_bytes=len(data))
    result = dict(schema_version='adaptability-audit-archive-v1', role='development', core_confirmed=False,
        source_directory=str(source), complete_matrix_verified=True, records=len(rows), probes=len(probes),
        compressed_artifacts=compressed, per_snr=analyze(spec,rows,probes),
        interval_interpretation='3个独立信道seed的逐SNR配对t区间，仅探索性、未作多重比较校正；不作为最终显著性确认。',
        harmful_frame_interpretation='接受帧上N劣于P是状态配对描述，不等于该次更新相对更新前有害；当前日志未保存更新前Data BER。')
    (destination/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',required=True)
    parser.add_argument('--destination',required=True)
    args = parser.parse_args()
    result=archive(args.source,args.destination)
    print(json.dumps(dict(records=result['records'],probes=result['probes'],complete_matrix_verified=True)))
