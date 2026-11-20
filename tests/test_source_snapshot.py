# -*- coding: utf-8 -*-
"""保证换行、源码污染或错误实验身份不会破坏运行指纹审计。"""

import copy
import hashlib
import json

import pytest

from evaluation.source_snapshot import snapshot_fingerprints, verify_source_snapshot


def _fixture(tmp_path):
    contents = {'receiver.py': '# 中文注释\r\nprint(1)\r\n',
                'config.json': '{}\n', 'manifest.json': '{}\r\n', 'dynamics.json': '{"rho":0.8}\n'}
    digests = {name: hashlib.sha256(value.encode('utf8')).hexdigest() for name, value in contents.items()}
    protocol = dict(experiment='example', git_commit='commit', sources={'receiver.py': digests['receiver.py']},
        config='config.json', config_sha256=digests['config.json'], manifest='manifest.json',
        manifest_sha256=digests['manifest.json'], variants=[dict(dynamics_file='dynamics.json',
            dynamics_file_sha256=digests['dynamics.json'])])
    payload = dict(schema_version=1, experiment='example', run_git_commit='commit', protocol=copy.deepcopy(protocol),
        files={name: dict(sha256=digests[name], content_utf8=content) for name, content in contents.items()})
    path = tmp_path / 'source_snapshot.json'
    path.write_bytes(json.dumps(payload, ensure_ascii=False, indent=2).encode('utf8'))
    return path, payload, protocol


def test_snapshot_preserves_mixed_line_endings_even_if_json_whitespace_is_converted(tmp_path):
    path, _, protocol = _fixture(tmp_path)
    path.write_bytes(path.read_bytes().replace(b'\n', b'\r\n'))
    snapshot = verify_source_snapshot(path, protocol)
    assert snapshot['files']['receiver.py']['content_utf8'] == '# 中文注释\r\nprint(1)\r\n'
    assert len(snapshot_fingerprints(protocol)) == 4


@pytest.mark.parametrize('corruption', ['source_content', 'source_digest', 'missing_source', 'extra_source', 'protocol', 'experiment'])
def test_incomplete_changed_or_wrong_experiment_snapshot_is_rejected(tmp_path, corruption):
    path, payload, protocol = _fixture(tmp_path)
    if corruption == 'source_content':
        payload['files']['receiver.py']['content_utf8'] = 'print(2)\n'
    elif corruption == 'source_digest':
        payload['files']['receiver.py']['sha256'] = 'wrong'
    elif corruption == 'missing_source':
        del payload['files']['receiver.py']
    elif corruption == 'extra_source':
        payload['files']['extra.py'] = dict(sha256='x', content_utf8='')
    elif corruption == 'protocol':
        payload['protocol']['git_commit'] = 'other'
    else:
        payload['experiment'] = 'other'
    path.write_bytes(json.dumps(payload, ensure_ascii=False).encode('utf8'))
    with pytest.raises(ValueError):
        verify_source_snapshot(path, protocol)
