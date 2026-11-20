# -*- coding: utf-8 -*-
"""只读审计完整 schema 3 实验，并按预登记窗口汇总；不重跑接收轨迹。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.receiver_diagnostics import LONG_DIAGNOSTIC_FIELDS, RECEIVER_SCORE_FIELDS
from scripts.compare_continual_variants import compact_stream_row


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(result_path):
    """完整验证后才生成性能汇总，缺失或失败实验不能作为完整结果。"""
    result_path = Path(result_path)
    result = json.loads(result_path.read_text(encoding='utf8'))
    require(result.get('status') == 'completed', '只汇总全部完成的实验。')
    protocol = result['protocol']
    manifest_path = ROOT / protocol['manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf8'))
    require(protocol['stream_rows_schema_version'] == 3 and protocol['long_diagnostics'], '需要显式 schema 3。')
    for field in ('seeds', 'snrs', 'frames', 'state_split', 'device'):
        require(protocol[field] == manifest[field], f'运行与预登记不匹配：{field}')
    require(protocol['manifest_sha256'] == sha(manifest_path), '协议指纹不匹配。')
    require(protocol['config_sha256'] == sha(ROOT / protocol['config']), '配置指纹不匹配。')
    for path, digest in protocol['checkpoint_sha256'].items():
        require(sha(path) == digest, 'checkpoint 指纹不匹配。')
    for path, digest in protocol['sources'].items():
        require(sha(ROOT / path) == digest, f'当前算法源码与运行版本不匹配：{path}')
    for variant in protocol['variants']:
        if variant.get('dynamics_file'):
            require(sha(ROOT / variant['dynamics_file']) == variant['dynamics_file_sha256'], '动态参数指纹不匹配。')
    arms = protocol['arms']
    require(arms == [variant['name'] for variant in manifest['variants']], '臂顺序不匹配。')
    variants = {row['name']: row for row in protocol['variants']}
    frames = protocol['frames']
    windows = manifest['pre_registered_windows']
    require(windows and windows[0][0] == 0 and windows[-1][1] == frames, '窗口必须覆盖首帧到末帧。')
    require(all(0 <= a < b <= frames for a, b in windows), '窗口边界非法。')
    require(all(windows[i][1] == windows[i + 1][0] for i in range(len(windows) - 1)), '窗口必须连续且不重叠。')
    frame_windows = {}
    for index, (start, end) in enumerate(windows):
        for frame in range(start, end):
            frame_windows[frame] = index
    sessions = [(snr, seed) for snr in protocol['snrs'] for seed in protocol['seeds']]
    acquisition = {(row['snr_db'], row['seed']): row for row in result['acquisitions']}
    require(len(result['acquisitions']) == len(acquisition) == len(sessions), 'acquisition 不齐或重复。')
    require(set(acquisition) == set(sessions), 'acquisition 会话不匹配。')
    config = json.loads((ROOT / protocol['config']).read_text(encoding='utf8'))
    expected_bits = config['frame_len'] - config['pilot_total']
    expected_observations = config['pilot_total'] - config['reward_pilot_total'] - config['max_delay']
    metric_fields = ('elapsed_ms', 'pilot_innovation_mse', 'pilot_innovation_variance_mean',
        'prior_covariance_trace', 'decode_covariance_trace', 'posterior_covariance_trace',
        'noise_variance_used') + RECEIVER_SCORE_FIELDS + LONG_DIAGNOSTIC_FIELDS
    accumulated = defaultdict(lambda: dict(n=0, errors=0, bits=0, sums=defaultdict(float)))
    rows_path = result_path.parent / result['rows_file']
    expected_count = len(sessions) * frames * len(arms)
    paired_hash = None
    paired_truth = None
    row_count = 0
    common_fields = ('true_cir_norm_diagnostic', 'true_cir_power_diagnostic',
        'true_frequency_flatness_diagnostic', 'true_data_signal_power_diagnostic',
        'realized_data_noise_power_diagnostic', 'true_effective_esn0_db_diagnostic', 'rx_power')
    with rows_path.open(encoding='utf8') as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            compact_stream_row(row, schema_version=3)
            require(index < expected_count, '出现多余记录。')
            session_index, within = divmod(index, frames * len(arms))
            frame_index, arm_index = divmod(within, len(arms))
            snr, seed = sessions[session_index]
            arm = arms[arm_index]
            require((row['snr_db'], row['seed'], row['frame_index'], row['arm']) ==
                    (snr, seed, frame_index, arm), f'记录顺序、缺失或重复：行{index + 1}')
            require(row['bits'] == expected_bits and type(row['errors']) is int and
                    0 <= row['errors'] <= expected_bits, 'Data 比特计数非法。')
            require(row['pilot_observations'] == expected_observations, 'Adapt 观测行数错误。')
            require(row['noise_variance_used'] == acquisition[(snr, seed)]['noise_estimate'], '噪声来源不匹配。')
            for field in metric_fields:
                require(isinstance(row[field], (int, float)) and math.isfinite(row[field]), f'诊断不是有限标量：{field}')
            for field in ('elapsed_ms', 'pilot_innovation_mse', 'pilot_innovation_variance_mean',
                          'prior_covariance_trace', 'decode_covariance_trace', 'posterior_covariance_trace',
                          'data_bce', 'data_brier', 'decoder_signal_mismatch_nmse_diagnostic'):
                require(row[field] >= 0, f'诊断非法负值：{field}')
            require(0 <= row['confident_wrong_data_count'] <= row['confident_data_count'] <= expected_bits,
                    '高置信比特计数非法。')
            require(0 <= row['data_brier'] <= 1 and 0 < row['true_frequency_flatness_diagnostic'] <= 1.00001,
                    '概率或频谱平坦度越界。')
            require(row['model_calls'] == 1, '每帧完整解码调用数不为1。')
            variant = variants[arm]
            require(row['neural_forwards'] == int(variant['decoder'] in {'neural', 'known_adapt_real_neural'}),
                    '神经前向计数不匹配。')
            require(row['pilot_updates'] == int(variant['arm'] != 'acquisition') and
                    row['data_updates'] == int(variant['arm'] == 'recursive_soft'), '在线更新计数不匹配。')
            if not row['data_updates']:
                require(row['decode_covariance_trace'] == row['posterior_covariance_trace'], '无Data更新却改变协方差。')
            require(row['decode_covariance_trace'] <= row['prior_covariance_trace'] + 1e-6, 'Pilot 更新后协方差增大。')
            truth_values = tuple(row[field] for field in common_fields)
            if arm_index == 0:
                paired_hash, paired_truth = row['rx_sha256'], truth_values
                require(isinstance(paired_hash, str) and len(paired_hash) == 64, '波形哈希非法。')
            else:
                require(row['rx_sha256'] == paired_hash and truth_values == paired_truth, '同帧波形或物理评分不一致。')
            for window_index in (frame_windows[frame_index], -1):
                item = accumulated[(snr, seed, arm, window_index)]
                item['n'] += 1
                item['errors'] += row['errors']
                item['bits'] += row['bits']
                for field in metric_fields:
                    item['sums'][field] += row[field]
            row_count += 1
    require(row_count == expected_count == result['row_count'], '总记录数不匹配或未完成。')
    session_summary = []
    for (snr, seed, arm, window_index), item in sorted(accumulated.items()):
        bounds = [0, frames] if window_index == -1 else windows[window_index]
        require(item['n'] == bounds[1] - bounds[0], '窗口记录数不匹配。')
        session_summary.append(dict(snr_db=snr, seed=seed, arm=arm, window=bounds,
            scope='whole_session' if window_index == -1 else 'registered_window',
            frames=item['n'], errors=item['errors'], bits=item['bits'], ber=item['errors'] / item['bits'],
            means={field: item['sums'][field] / item['n'] for field in metric_fields}))
    return dict(status='completed_development_audit', schema_version=3, sessions=len(sessions),
        rows=row_count, paired_frames=len(sessions) * frames, session_arm_groups=len(sessions) * len(arms),
        duplicates=0, missing=0, hash_mismatches=0, invalid_fields=0,
        result_sha256=sha(result_path), rows_sha256=sha(rows_path), manifest_sha256=sha(manifest_path),
        interpretation='逐会话、预登记窗口的开发描述；不作总体确认、等价、饱和或机制因果归因。',
        session_summary=session_summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    args = parser.parse_args()
    result_path = Path(args.result)
    summary = audit(result_path)
    output = result_path.with_name(result_path.stem + '_audit.json')
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(f"完整审计通过：{summary['sessions']}会话，{summary['rows']}行，{summary['paired_frames']}个配对帧。")


if __name__ == '__main__':
    main()
