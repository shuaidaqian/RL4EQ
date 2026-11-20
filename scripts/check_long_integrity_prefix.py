# -*- coding: utf-8 -*-
"""只审计运行中schema3流的固定字节前缀，不读取主JSON、不汇总性能。"""

import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_continual_variants import compact_stream_row


def inspect_prefix(rows_path, manifest, config):
    """校验顺序、计数、字段与同帧配对，只返回完整性信息。

    文件继续增长不改变本次审计范围。末尾未写完的一行留待下次，不把
    临时半行当故障；已经完整换行的非法JSON、重复或缺帧则立即拒绝。
    """

    rows_path = Path(rows_path)
    size = rows_path.stat().st_size
    variants = manifest['variants']
    arms = [row['name'] for row in variants]
    sessions = [(snr, seed) for snr in manifest['snrs'] for seed in manifest['seeds']]
    frames = manifest['frames']
    expected_bits = config['frame_len'] - config['pilot_total']
    expected_observations = config['pilot_total'] - config['reward_pilot_total'] - config['max_delay']
    common_fields = ('true_cir_norm_diagnostic', 'true_cir_power_diagnostic',
        'true_frequency_flatness_diagnostic', 'true_data_signal_power_diagnostic',
        'realized_data_noise_power_diagnostic', 'true_effective_esn0_db_diagnostic', 'rx_power')
    count, end_offset = 0, 0
    completed, session_noise = [], {}
    last = None
    paired_hash, paired_truth = None, None
    digest = hashlib.sha256()
    with rows_path.open('rb') as stream:
        while stream.tell() < size:
            line = stream.readline(size - stream.tell())
            if not line.endswith(b'\n'):
                break
            row = json.loads(line)
            checked = compact_stream_row(row, schema_version=3)
            for key, value in checked.items():
                if key in {'arm', 'rx_sha256', 'shared_state_from'}:
                    continue
                if type(value) not in {int, float} or not math.isfinite(value):
                    raise ValueError(f'完整行出现非有限数值或非数值诊断：{key}')
            session_index, within = divmod(count, frames * len(arms))
            frame_index, arm_index = divmod(within, len(arms))
            if session_index >= len(sessions):
                raise ValueError('流文件存在超出预登记总量的记录。')
            snr, seed = sessions[session_index]
            expected = (snr, seed, frame_index, arms[arm_index])
            actual = tuple(row[key] for key in ('snr_db', 'seed', 'frame_index', 'arm'))
            if actual != expected:
                raise ValueError(f'帧顺序、缺失或重复异常：行{count + 1}')
            integer_fields = ('snr_db', 'seed', 'frame_index', 'errors', 'bits', 'pilot_observations',
                              'model_calls', 'neural_forwards', 'pilot_updates', 'data_updates',
                              'confident_data_count', 'confident_wrong_data_count')
            if any(type(row[key]) is not int for key in integer_fields):
                raise ValueError('索引或计数字段不是整数。')
            if (not 0 <= row['errors'] <= expected_bits or
                    row['bits'] != expected_bits or row['pilot_observations'] != expected_observations):
                raise ValueError('比特计数或Adapt观测行数非法。')
            variant = variants[arm_index]
            expected_neural = int(variant.get('decoder', 'neural') in {'neural', 'known_adapt_real_neural'})
            if (row['model_calls'] != 1 or row['neural_forwards'] != expected_neural or
                    row['pilot_updates'] != int(variant['arm'] != 'acquisition') or
                    row['data_updates'] != int(variant['arm'] == 'recursive_soft')):
                raise ValueError('前向或更新计数不匹配预登记臂。')
            if not row['data_updates'] and row['decode_covariance_trace'] != row['posterior_covariance_trace']:
                raise ValueError('没有Data更新却改变了解码后的协方差。')
            if row['decode_covariance_trace'] > row['prior_covariance_trace'] + 1e-6:
                raise ValueError('Pilot更新后协方差迹增大。')
            if not 0 <= row['confident_wrong_data_count'] <= row['confident_data_count'] <= expected_bits:
                raise ValueError('高置信比特计数非法。')
            if row['noise_variance_used'] <= 0:
                raise ValueError('噪声功率必须为正。')
            session_noise.setdefault((snr, seed), row['noise_variance_used'])
            if row['noise_variance_used'] != session_noise[(snr, seed)]:
                raise ValueError('同会话共享acquisition噪声不一致。')
            truth_values = tuple(row[key] for key in common_fields)
            if arm_index == 0:
                paired_hash, paired_truth = row['rx_sha256'], truth_values
                if not isinstance(paired_hash, str) or len(paired_hash) != 64:
                    raise ValueError('波形哈希非法。')
            elif row['rx_sha256'] != paired_hash or truth_values != paired_truth:
                raise ValueError('同帧波形或物理评分不一致。')
            if frame_index == frames - 1 and arm_index == len(arms) - 1:
                completed.append(dict(snr_db=snr, seed=seed, arms=len(arms),
                    rows=frames * len(arms), paired_frames=frames))
            digest.update(line)
            end_offset = stream.tell()
            count += 1
            last = {key: row[key] for key in ('snr_db', 'seed', 'frame_index', 'arm')}
    return dict(role='integrity_only_no_performance_analysis',
        verified_at_local=datetime.datetime.now().astimezone().isoformat(timespec='seconds'),
        file_size_at_snapshot=size, audited_byte_prefix=end_offset,
        trailing_incomplete_line_bytes=size - end_offset, prefix_sha256=digest.hexdigest(),
        rows=count, paired_frames=count // len(arms), complete_sessions=completed, last=last,
        all_expected_rows_saved=count == frames * len(arms) * len(sessions),
        duplicates=0, missing_within_saved_prefix=0, hash_mismatches=0,
        invalid_counts_or_checked_schema3_fields=0,
        interpretation='只说明保存前缀完整性，不证明进程存活、主结果completed或研究性能。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', required=True)
    parser.add_argument('--source-snapshot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    snapshot_path = Path(args.source_snapshot)
    snapshot = json.loads(snapshot_path.read_bytes())
    from evaluation.source_snapshot import verify_source_snapshot
    verify_source_snapshot(snapshot_path, snapshot['protocol'])
    protocol = snapshot['protocol']
    manifest = json.loads(snapshot['files'][protocol['manifest']]['content_utf8'])
    config = json.loads(snapshot['files'][protocol['config']]['content_utf8'])
    checkpoint = inspect_prefix(args.rows, manifest, config)
    checkpoint['source_snapshot_sha256'] = hashlib.sha256(snapshot_path.read_bytes()).hexdigest()
    checkpoint['run_git_commit'] = protocol['git_commit']
    Path(args.output).write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(f"完整性核验通过：{len(checkpoint['complete_sessions'])}个完整会话，"
          f"{checkpoint['rows']}行；未生成性能汇总。")


if __name__ == '__main__':
    main()
