# -*- coding: utf-8 -*-
"""核验运行源码的原始 UTF-8 字节归档，不执行归档内代码。"""

import hashlib
import json
from pathlib import Path


def snapshot_fingerprints(protocol):
    expected = dict(protocol['sources'])
    expected[protocol['config']] = protocol['config_sha256']
    expected[protocol['manifest']] = protocol['manifest_sha256']
    for variant in protocol['variants']:
        if variant.get('dynamics_file'):
            expected[variant['dynamics_file']] = variant['dynamics_file_sha256']
    return expected


def verify_source_snapshot(path, protocol):
    """逐字节校验所有源码和输入文件，协议必须与待分析实验完全一致。

    JSON 换行转换不会改变 content_utf8 字符串内转义保存的原始换行；
    重新编码后必须匹配启动指纹。归档仅用于审计，不导入或执行其代码。
    """

    payload = json.loads(Path(path).read_bytes())
    if payload.get('schema_version') != 1 or payload.get('protocol') != protocol:
        raise ValueError('源码快照版本或完整运行协议不匹配。')
    if payload.get('experiment') != protocol['experiment'] or payload.get('run_git_commit') != protocol['git_commit']:
        raise ValueError('源码快照实验身份不匹配。')
    expected = snapshot_fingerprints(protocol)
    if set(payload.get('files', {})) != set(expected):
        raise ValueError('源码快照文件清单不完整或有额外文件。')
    for name, digest in expected.items():
        item = payload['files'][name]
        raw = item['content_utf8'].encode('utf8')
        if item['sha256'] != digest or hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'源码快照原始字节指纹不匹配：{name}')
    return payload
