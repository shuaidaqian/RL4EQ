# -*- coding: utf-8 -*-
"""只读统计GN验证Pilot门控变化与动作后Data硬BER变化的对应关系。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def diagnose(metrics_path: Path, output_path: Path):
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    grouped = []
    for snr in sorted({int(row["snr_db"]) for row in rows}):
        arms = {}
        for arm in sorted({row["training"] for row in rows}):
            active = [row for row in rows if int(row["snr_db"]) == snr and row["training"] == arm]
            accepted = [row for row in active if row["accepted"]]
            improved = sum(row["ber_online"] < row["ber_before"] for row in accepted)
            unchanged = sum(row["ber_online"] == row["ber_before"] for row in accepted)
            worsened = sum(row["ber_online"] > row["ber_before"] for row in accepted)
            mean_validation_gain = (sum(row["validation_loss_before"] - row["validation_loss_retained"]
                                        for row in accepted) / len(accepted) if accepted else 0.)
            arms[arm] = dict(frames=len(active), candidate_accepted=sum(row["candidate_accepted"] for row in active),
                accepted=len(accepted), reset_to_origin=sum(row["reset_to_origin"] for row in active),
                accepted_data_ber_improved=improved, accepted_data_ber_unchanged=unchanged,
                accepted_data_ber_worsened=worsened,
                accepted_fraction_without_data_ber_improvement=(
                    (unchanged + worsened) / len(accepted) if accepted else None),
                mean_accepted_validation_loss_reduction=mean_validation_gain)
        grouped.append(dict(snr_db=snr, arms=arms))
    result = dict(role="read_only_posthoc_alignment_diagnostic",
        online_control_used_data_labels=False,
        interpretation="Data BER只在所有候选、守门和回退完成后评分；此统计不修改任何更新或门控动作。",
        source_metrics_path=str(metrics_path), source_metrics_sha256=sha256(metrics_path),
        records=len(rows), per_snr=grouped)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(diagnose(Path(args.metrics), Path(args.output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
