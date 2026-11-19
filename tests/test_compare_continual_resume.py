import json

import pytest

from scripts.compare_continual_variants import inspect_stream_for_resume


def _write_rows(path, *, frames=4, partial_at=2, corrupt_hash=False, duplicate=False):
    arms = ('a', 'b')
    rows = []
    for seed, end in ((1, frames), (2, partial_at)):
        for frame in range(end):
            for arm in arms:
                digest = f'{seed}-{frame}'
                if corrupt_hash and seed == 2 and frame == 1 and arm == 'b':
                    digest = 'wrong'
                rows.append(dict(snr_db=0, seed=seed, frame_index=frame,
                                 arm=arm, errors=frame, bits=10, rx_sha256=digest))
    if duplicate:
        rows.append(rows[-1])
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    return arms, rows


def test_resume_stream_accepts_complete_sessions_and_one_matched_prefix(tmp_path):
    path = tmp_path / 'rows.jsonl'
    arms, rows = _write_rows(path)

    complete, partial_key, partial_rows, count = inspect_stream_for_resume(
        path, snrs=[0], seeds=[1, 2], arms=arms, frames=4)

    assert complete == {(0, 1)}
    assert partial_key == (0, 2)
    assert set(partial_rows['a']) == {0, 1}
    assert count == len(rows) == 12


def test_resume_stream_rejects_duplicate_rows(tmp_path):
    path = tmp_path / 'rows.jsonl'
    arms, _ = _write_rows(path, duplicate=True)

    with pytest.raises(ValueError, match='重复记录'):
        inspect_stream_for_resume(path, snrs=[0], seeds=[1, 2], arms=arms, frames=4)


def test_resume_stream_rejects_mismatched_prefix_waveforms(tmp_path):
    path = tmp_path / 'rows.jsonl'
    arms, _ = _write_rows(path, corrupt_hash=True)

    with pytest.raises(ValueError, match='部分会话的配对波形不一致'):
        inspect_stream_for_resume(path, snrs=[0], seeds=[1, 2], arms=arms, frames=4)
