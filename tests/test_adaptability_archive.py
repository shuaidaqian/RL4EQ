# -*- coding: utf-8 -*-
"""拒绝缺失、重复、信息越界与探针错配的开发结果。"""

import copy
import gzip
import json

import pytest

from scripts.summarize_adaptability_audit import archive, digest, validate_records


def fixture():
    spec=dict(seeds=[1],snrs_db=[0],frames=2,fixed_candidate_offsets=[0,1,128],
        deterministic_update=dict(total_radius=.8))
    summary=dict(smoke=False,core_confirmed=False,role='development',seeds=[1],snrs_db=[0],frames_per_seed_snr=2)
    rows=[dict(seed=1,snr_db=0,frame=i,ber_F=.2,ber_P=.15,ber_N=.14,ber_T=.1,ber_N_candidate=.13,
        retained_total_norm=.2,uses_rl=False,data_labels_used_for_control=False,
        shared_physical_context_for_P_N=True,rx_sha256=str(i),lookahead_rx_sha256='future'+str(i)) for i in range(2)]
    probes=[dict(seed=1,snr_db=0,offset=i,source_frame=0,future_metrics_used_for_control=False,
        ber_P=.15,ber_raw=.13,ber_guarded=.14,
        rx_sha256=str(i),lookahead_rx_sha256='future'+str(i)) for i in [0,1,128]]
    return spec,summary,rows,probes


def test_complete_development_coverage_validates():
    assert validate_records(*fixture())


@pytest.mark.parametrize('corruption', ['missing_frame','duplicate_frame','missing_probe','duplicate_probe',
    'waveform','labels','radius','nan','formal','smoke','wrong_budget','probe_nan'])
def test_invalid_development_results_are_rejected(corruption):
    spec,summary,rows,probes=fixture()
    if corruption=='missing_frame': rows.pop()
    if corruption=='duplicate_frame': rows.append(copy.deepcopy(rows[0]))
    if corruption=='missing_probe': probes.pop()
    if corruption=='duplicate_probe': probes.append(copy.deepcopy(probes[0]))
    if corruption=='waveform': probes[0]['rx_sha256']='wrong'
    if corruption=='labels': rows[0]['data_labels_used_for_control']=True
    if corruption=='radius': rows[0]['retained_total_norm']=1.
    if corruption=='nan': rows[0]['ber_N']=float('nan')
    if corruption=='formal': summary['core_confirmed']=True
    if corruption=='smoke': summary['smoke']=True
    if corruption=='wrong_budget': summary['frames_per_seed_snr']=3
    if corruption=='probe_nan': probes[0]['ber_raw']=float('nan')
    with pytest.raises(ValueError):
        validate_records(spec,summary,rows,probes)


def test_archive_preserves_full_raw_bytes_and_rejects_tampering(tmp_path,monkeypatch):
    import scripts.summarize_adaptability_audit as module
    spec,summary,rows,probes=fixture()
    raw=('\n'.join(json.dumps(row) for row in rows)+'\n').encode()
    probe_raw=('\n'.join(json.dumps(row) for row in probes)+'\n').encode()
    summary.update(records=len(rows),fixed_candidate_records=len(probes),raw_sha256=digest(raw),
        fixed_candidate_raw_sha256=digest(probe_raw))
    source=tmp_path/'source'
    source.mkdir()
    (source/'manifest.json').write_text(json.dumps(spec),encoding='utf-8')
    (source/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    (source/'frame_metrics.jsonl').write_bytes(raw)
    (source/'fixed_candidate_metrics.jsonl').write_bytes(probe_raw)
    monkeypatch.setattr(module,'analyze',lambda *args: [])
    result=archive(source,tmp_path/'archive')
    assert result['complete_matrix_verified']
    assert gzip.decompress((tmp_path/'archive/frame_metrics.jsonl.gz').read_bytes())==raw
    with pytest.raises(ValueError,match='禁止覆盖'):
        archive(source,tmp_path/'archive')
    (source/'frame_metrics.jsonl').write_bytes(raw+b'\n')
    with pytest.raises(ValueError,match='SHA256'):
        archive(source,tmp_path/'tampered')
