# -*- coding: utf-8 -*-
"""仅用 CPU 准备确定性的离线长轨迹样本，不训练权重、不访问测试分布。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.pilot_transfer import PilotTransferTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--sample-start", type=int, required=True)
    parser.add_argument("--sample-stop", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if not config.get("pilot_transfer", {}).get("offline_sample_cache_dir") or not 0 <= args.sample_start < args.sample_stop:
        raise ValueError("需要显式离线缓存和有效的半开样本索引范围。")
    trainer = PilotTransferTrainer(config, "cpu", objective="baseline")
    started = time.monotonic()
    for index in range(args.sample_start, args.sample_stop):
        trainer.sample_index = index
        _, _, snr, seed = trainer._sample()
        print(json.dumps({"sample_index": index, "seed": seed, "snr_db": snr,
            "cache_key": trainer.last_sample_cache_key, "elapsed_seconds": time.monotonic() - started}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
