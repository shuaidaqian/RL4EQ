# -*- coding: utf-8 -*-
"""运行中前缀审计不误读半行、不掩盖缺失和错配、不汇总中途性能。"""

import json

import pytest

from evaluation.receiver_diagnostics import LONG_DIAGNOSTIC_FIELDS, RECEIVER_SCORE_FIELDS
from scripts.compare_continual_variants import STREAM_ROW_FIELDS
from scripts.check_long_integrity_prefix import inspect_prefix


def _fixture(tmp_path):
    manifest = dict(variants=[dict(name='f', arm='acquisition', decoder='known_adapt_real_physics'),
        dict(name='o', arm='recursive_pilot', decoder='known_adapt_real_physics')],
        snrs=[0], seeds=[1, 2], frames=2)
    config = dict(frame_len=128, pilot_total=64, reward_pilot_total=16, max_delay=2)
    rows = []
    for seed, stop in [(1, 2), (2, 1)]:
        for frame in range(stop):
            for arm in ['f', 'o']:
                row = dict.fromkeys(STREAM_ROW_FIELDS + RECEIVER_SCORE_FIELDS + LONG_DIAGNOSTIC_FIELDS, 1.)
                row.update(seed=seed, snr_db=0, frame_index=frame, arm=arm, errors=3, bits=64,
                    rx_sha256='a' * 64, pilot_observations=46, neural_forwards=0,
                    pilot_updates=int(arm == 'o'), data_updates=0, model_calls=1,
                    confident_data_count=1, confident_wrong_data_count=1)
                rows.append(row)
    path = tmp_path / 'rows.jsonl'
    return path, rows, manifest, config


def _write(path, rows, suffix=b''):
    path.write_bytes((''.join(json.dumps(row) + '\n' for row in rows)).encode('utf8') + suffix)


def test_prefix_audit_ignores_only_unfinished_last_line_and_never_reports_performance(tmp_path):
    path, rows, manifest, config = _fixture(tmp_path)
    _write(path, rows, b'{"unfinished":')
    result = inspect_prefix(path, manifest, config)
    assert result['rows'] == 6
    assert result['paired_frames'] == 3
    assert len(result['complete_sessions']) == 1
    assert result['trailing_incomplete_line_bytes'] == len(b'{"unfinished":')
    assert result['all_expected_rows_saved'] is False
    assert 'ber' not in result and 'total_errors' not in result and 'status' not in result


@pytest.mark.parametrize('corrupt', ['duplicate', 'missing', 'hash', 'count', 'nan', 'physical',
                                  'null_diagnostic', 'boolean_diagnostic', 'fractional_index'])
def test_complete_invalid_records_are_rejected_instead_of_treated_as_partial(tmp_path, corrupt):
    path, rows, manifest, config = _fixture(tmp_path)
    if corrupt == 'duplicate':
        rows.insert(1, dict(rows[0]))
    elif corrupt == 'missing':
        rows.pop(1)
    elif corrupt == 'hash':
        rows[1]['rx_sha256'] = 'b' * 64
    elif corrupt == 'count':
        rows[1]['errors'] = 65
    elif corrupt == 'nan':
        rows[1]['decoder_signal_mismatch_nmse_diagnostic'] = float('nan')
    elif corrupt == 'physical':
        rows[1]['true_cir_power_diagnostic'] = 2.
    elif corrupt == 'null_diagnostic':
        rows[1]['decoder_signal_mismatch_nmse_diagnostic'] = None
    elif corrupt == 'boolean_diagnostic':
        rows[1]['decoder_signal_mismatch_nmse_diagnostic'] = True
    else:
        rows[1]['frame_index'] = 0.
    _write(path, rows)
    with pytest.raises(ValueError):
        inspect_prefix(path, manifest, config)
