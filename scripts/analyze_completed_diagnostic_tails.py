# -*- coding: utf-8 -*-
"""只读完整开发结果，描述逐帧诊断尾部与错误集中度，不生成研究会话。"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.long_diagnostic_summary import derived_diagnostics


METRICS = {
    'cir_phase_only_squared_error_diagnostic': 'high',
    'decoder_phase_data_rms_rad_diagnostic': 'high',
    'decoder_signal_mismatch_nmse_diagnostic': 'high',
    'true_cir_power_diagnostic': 'low',
    'true_frequency_flatness_diagnostic': 'low',
}


def summarize_distribution(values, errors, bits, *, bad_direction):
    """保留实际分位点与入选帧数，阈值并列时不强行凑足固定比例。"""
    values = np.asarray(values, dtype=float)
    errors = np.asarray(errors, dtype=np.int64)
    bits = np.asarray(bits, dtype=np.int64)
    if (not len(values) or len(values) != len(errors) or len(errors) != len(bits)
            or not np.isfinite(values).all() or (errors < 0).any() or (bits <= 0).any()
            or (errors > bits).any() or bad_direction not in {'high', 'low'}):
        raise ValueError('分布诊断需要完整有限标量与合法错误/比特计数。')
    total_errors = int(errors.sum())
    result = dict(frames=len(values), mean=float(values.mean()), min=float(values.min()),
        max=float(values.max()), quantiles={str(q): float(np.quantile(values, q)) for q in (.5, .9, .95, .99)},
        total_errors=total_errors, ber=float(errors.sum() / bits.sum()), tails=[])
    # 低于20帧的窗口不赋予5%/1%尾部统计含义，仍完整保留其均值与分位数。
    if len(values) < 20:
        return result
    for fraction in (.1, .05, .01):
        quantile = 1 - fraction if bad_direction == 'high' else fraction
        threshold = float(np.quantile(values, quantile))
        selected = values >= threshold if bad_direction == 'high' else values <= threshold
        selected_errors = int(errors[selected].sum())
        selected_bits = int(bits[selected].sum())
        result['tails'].append(dict(nominal_fraction=fraction, threshold=threshold,
            frames=int(selected.sum()), frame_fraction=float(selected.mean()), errors=selected_errors,
            error_fraction=selected_errors / total_errors if total_errors else None,
            conditional_ber=selected_errors / selected_bits))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--arm', default='recursive_pilot_ar1')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    result = json.loads(result_path.read_bytes())
    audit_path = result_path.with_name(result_path.stem + '_audit.json')
    audit = json.loads(audit_path.read_bytes())
    digest = hashlib.sha256(result_path.read_bytes()).hexdigest()
    if (result.get('status') != 'completed' or audit.get('status') != 'completed_development_audit'
            or audit['result_sha256'] != digest):
        raise ValueError('只允许分析已完成且有匹配完整审计的结果。')
    manifest = audit['manifest']
    if args.arm not in result['protocol']['arms']:
        raise ValueError('诊断臂不在原实验中。')
    windows = manifest['pre_registered_windows']
    sessions = {(snr, seed) for snr in manifest['snrs'] for seed in manifest['seeds']}
    groups = {(snr, seed, tuple(window)): [] for snr, seed in sessions for window in windows}
    frame_window = {frame: tuple(window) for window in windows for frame in range(*window)}
    rows_path = result_path.parent / result['rows_file']
    rows_digest = hashlib.sha256()
    with rows_path.open('rb') as stream:
        for raw in stream:
            rows_digest.update(raw)
            row = json.loads(raw)
            if row['arm'] != args.arm:
                continue
            row.update(derived_diagnostics(row))
            group = groups[(row['snr_db'], row['seed'], frame_window[row['frame_index']])]
            group.append(tuple(row[key] for key in METRICS) + (row['errors'], row['bits']))
    if rows_digest.hexdigest() != audit['rows_sha256']:
        raise ValueError('原始流指纹与完整审计不匹配。')
    summaries = []
    for (snr, seed, window), group in sorted(groups.items()):
        if len(group) != window[1] - window[0]:
            raise ValueError('逐会话原窗口不齐，拒绝选择性报告。')
        values = np.asarray(group)
        summaries.append(dict(snr_db=snr, seed=seed, window=list(window),
            metrics={key: summarize_distribution(values[:, index], values[:, -2], values[:, -1],
                bad_direction=direction) for index, (key, direction) in enumerate(METRICS.items())}))
    payload = dict(role='posthoc_completed_development_diagnostic_no_confirmation', arm=args.arm,
        result_sha256=digest, rows_sha256=rows_digest.hexdigest(),
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        analysis_plan='原实验所有预登记窗口、逐会话分别计算五项诊断的均值/范围和p50/p90/p95/p99；至少20帧时报告预设10%/5%/1%不利尾部的错误集中度。',
        limitations=['事后开发诊断，不是新增独立证据，不运行或复用研究seed',
            '同一会话的帧相关，不给独立帧p值或总体置信区间',
            '尾部可重叠，错误集中度不能相加分摊原因；相关性不等于因果',
            '阈值并列时保留全部并列帧，实际比例不强制等于名义比例',
            '不能由诊断未集中判定无影响或等价，也不据此更改原窗口与结论'],
        research_sessions_generated=0, summaries=summaries)
    output = Path(args.output)
    if output.exists():
        raise ValueError('分析输出已存在，拒绝隐式覆盖。')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(f'{len(summaries)}个原预登记会话窗口完成尾部描述；未运行研究会话，不作因果或总体判定。')


if __name__ == '__main__':
    main()
