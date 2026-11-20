# -*- coding: utf-8 -*-
"""从已完成逐帧评分派生诊断，不调用接收机或生成器。"""

import math
from collections import defaultdict

from scipy.special import ndtr


DERIVED_METRIC_FIELDS = (
    'cir_phase_only_squared_error_diagnostic',
    'pilot_innovation_energy_ratio_diagnostic',
    'full_energy_loose_genie_bound_diagnostic',
)


def derived_diagnostics(row):
    """先逐帧计算非线性量，再跨窗口平均，避免用平均值重建错误指标。

    下界使用事后仿真噪声和真实功率，只是理想辅助者的全能量宽松下界。
    未计算帧末截断；不得称为实际接收机或精确帧内下界。
    """

    truth_power = row['true_cir_power_diagnostic']
    true_noise = 10 ** (-row['snr_db'] / 10)
    predicted_variance = row['pilot_innovation_variance_mean']
    if truth_power < 0 or predicted_variance <= 0:
        raise ValueError('真实功率与预测能量不满足评分物理条件。')
    return dict(
        cir_phase_only_squared_error_diagnostic=row['phase_only_nmse_diagnostic'] * truth_power,
        pilot_innovation_energy_ratio_diagnostic=row['pilot_innovation_mse'] / predicted_variance,
        full_energy_loose_genie_bound_diagnostic=float(ndtr(-math.sqrt(2 * truth_power / true_noise))),
    )


def paired_window_summary(audit, manifest):
    """按既定会话和窗口配对，分别保留自身变化与相对优势变化。

    只作开发描述，不拟合单调形状，不搜索饱和起点，也不生成通过判定。
    """

    if audit.get('status') != 'completed_development_audit':
        raise ValueError('只允许分析完整审计通过的实验。')
    arms = [variant['name'] for variant in manifest['variants']]
    windows = [('whole_session', (0, manifest['frames']))] + [
        ('registered_window', tuple(window)) for window in manifest['pre_registered_windows']]
    lookup = {}
    for row in audit['session_summary']:
        key = (row['snr_db'], row['seed'], row['arm'], row['scope'], tuple(row['window']))
        if key in lookup:
            raise ValueError('会话窗口汇总存在重复记录。')
        lookup[key] = row
    expected = {(snr, seed, arm, scope, window)
                for snr in manifest['snrs'] for seed in manifest['seeds']
                for arm in arms for scope, window in windows}
    if set(lookup) != expected:
        raise ValueError('会话窗口汇总不完整或存在额外窗口，拒绝选择性配对。')
    grouped = defaultdict(list)
    for key, row in lookup.items():
        snr, seed, arm, scope, window = key
        if row['frames'] != window[1] - window[0] or row['bits'] <= 0:
            raise ValueError('汇总窗口帧数或比特数非法。')
        grouped[(snr, arm, scope, window)].append(row)
    aggregates = []
    for (snr, arm, scope, window), rows in sorted(grouped.items()):
        metric_names = set(rows[0]['means'])
        if any(set(row['means']) != metric_names for row in rows):
            raise ValueError('不同会话的诊断字段不一致。')
        # 包裹相位均值不能再做普通算术平均；保留在逐会话原始审计中。
        metric_names.discard('decoder_phase_data_mean_rad_diagnostic')
        aggregates.append(dict(snr_db=snr, arm=arm, scope=scope, window=list(window),
            sessions=len(rows), mean_ber=sum(row['ber'] for row in rows) / len(rows),
            total_errors=sum(row['errors'] for row in rows), total_bits=sum(row['bits'] for row in rows),
            means={name: sum(row['means'][name] for row in rows) / len(rows) for name in sorted(metric_names)}))
    paired = []
    changes = []
    for snr in manifest['snrs']:
        for seed in manifest['seeds']:
            for baseline, online in manifest['pairs']:
                for scope, window in windows:
                    baseline_row = lookup[(snr, seed, baseline, scope, window)]
                    online_row = lookup[(snr, seed, online, scope, window)]
                    if baseline_row['bits'] != online_row['bits']:
                        raise ValueError('配对两臂评分比特数不同。')
                    paired.append(dict(snr_db=snr, seed=seed, baseline=baseline, online=online,
                        scope=scope, window=list(window), baseline_ber=baseline_row['ber'],
                        online_ber=online_row['ber'], gain=baseline_row['ber'] - online_row['ber']))
                for earlier, later in zip(manifest['pre_registered_windows'], manifest['pre_registered_windows'][1:]):
                    previous_base = lookup[(snr, seed, baseline, 'registered_window', tuple(earlier))]['ber']
                    current_base = lookup[(snr, seed, baseline, 'registered_window', tuple(later))]['ber']
                    previous_online = lookup[(snr, seed, online, 'registered_window', tuple(earlier))]['ber']
                    current_online = lookup[(snr, seed, online, 'registered_window', tuple(later))]['ber']
                    delta_base = current_base - previous_base
                    delta_online = current_online - previous_online
                    changes.append(dict(snr_db=snr, seed=seed, baseline=baseline, online=online,
                        earlier_window=earlier, later_window=later,
                        delta_baseline_ber=delta_base, delta_online_ber=delta_online,
                        delta_gain=delta_base - delta_online))
    return dict(role='development_descriptive_no_confirmation',
        interpretation='会话等权；只用预登记窗口；自身BER变化与相对优势分别报告；未冻结实质/等价容差，不作通过或饱和判定。',
        aggregate_windows=aggregates, per_session_pairs=paired, adjacent_window_changes=changes)
