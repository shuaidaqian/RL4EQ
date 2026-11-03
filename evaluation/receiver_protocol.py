# -*- coding: utf-8 -*-
"""汇总前审计跨帧窗口，防止不同接收边界被合并成一项结果。"""

from collections import defaultdict


def audit_lookahead_rows(rows):
    """兼容历史无后缀记录；启用后缀时强制校验协议、配对 hash 和帧龄。"""
    protocols = set()
    pairs = defaultdict(set)
    for row in rows:
        length = row.get("lookahead_len", 0)
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise ValueError("跨帧窗口长度必须为非负整数。")
        version = row.get("lookahead_protocol_version", "none")
        future = row.get("lookahead_frame_index")
        signature = row.get("lookahead_rx_sha256")
        if length:
            if (version != "cross-frame-prefix-v1" or not isinstance(future, int)
                    or not isinstance(signature, str) or len(signature) != 64
                    or any(char not in "0123456789abcdef" for char in signature)):
                raise ValueError("跨帧记录缺少有效协议、帧号或接收波形 SHA256。")
            if "channel_frame_index" in row and future != row["channel_frame_index"] + 1:
                raise ValueError("跨帧窗口没有紧接当前物理帧。")
        elif version != "none" or future is not None or signature is not None:
            raise ValueError("无后缀协议不能携带跨帧观测。")
        protocols.add((length, version))
        key = (row["snr_db"], row["seed"], row["frame"])
        pairs[key].add((length, version, future, signature))
    if len(protocols) != 1 or any(len(values) != 1 for values in pairs.values()):
        raise ValueError("混合了不同跨帧协议或同帧后缀波形不一致。")
    length, version = next(iter(protocols))
    return {"lookahead_len": length, "lookahead_protocol_version": version,
            "shared_lookahead_rx_sha256_passed": bool(length)}
