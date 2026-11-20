# -*- coding: utf-8 -*-
"""核验并汇总长会话 AR(1) 开发实验的逐帧配对数据。"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import t


AUDITED_INITIAL_PREDICTION_OMISSION_SHA256 = (
    "4fcc2dee9a70c944c9ed9382d2218f4e9d2dc6e093127cad558b624337074b81"
)


def _mean_ci(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    half_width = float(t.ppf(0.975, n - 1)) * math.sqrt(variance / n)
    return [mean, mean - half_width, mean + half_width]


def summarize(result_path: Path, manifest_path: Path):
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError("实验尚未 completed，禁止生成最终汇总。")
    expected_rows = len(manifest["snrs"]) * len(manifest["seeds"]) * manifest["frames"] * len(manifest["variants"])
    if result.get("row_count") != expected_rows:
        raise ValueError("主 JSON 声明行数与预登记实验规模不符。")

    rows_path = result_path.parent / result["rows_file"]
    grouped = defaultdict(dict)
    row_count = 0
    with rows_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = (row["snr_db"], row["seed"], row["frame_index"])
            if row["arm"] in grouped[key]:
                raise ValueError(f"发现重复臂记录：{key} {row['arm']}")
            grouped[key][row["arm"]] = row
            row_count += 1
    if row_count != result["row_count"]:
        raise ValueError(f"JSONL 行数 {row_count} 与主 JSON 不匹配。")

    arms = [row["name"] for row in manifest["variants"]]
    seeds = manifest["seeds"]
    snrs = manifest["snrs"]
    frames = manifest["frames"]
    for snr in snrs:
        for seed in seeds:
            for index in range(frames):
                pair = grouped[(snr, seed, index)]
                if set(pair) != set(arms):
                    raise ValueError(f"臂不完整：SNR={snr}, seed={seed}, frame={index}")
                hashes = {pair[arm]["rx_sha256"] for arm in arms}
                if len(hashes) != 1:
                    raise ValueError(f"配对波形哈希不一致：SNR={snr}, seed={seed}, frame={index}")

    windows = [(0, 1), *[tuple(window) for window in manifest["pre_registered_windows"]]]
    pairs = [tuple(pair) for pair in manifest["pairs"]]
    # 使用实验记录的源码指纹归类，不能从当前工作树或 BER 曲线推断旧实现。
    receiver_sha256 = result["protocol"].get("sources", {}).get("training/continual_receiver.py")
    timing_audit = {"receiver_sha256": receiver_sha256, "status": "not_classified"}
    if receiver_sha256 == AUDITED_INITIAL_PREDICTION_OMISSION_SHA256:
        timing_audit.update({
            "status": "known_initial_prediction_omission",
            "evidence": "docs/continual_v2_initial_prediction_audit_20261120.md",
            "description": "已审计的接收器首个 Data Pilot 前跳过状态预测，第二个 Data 帧才开始预测。",
        })
    summary = {
        "experiment": result["protocol"]["experiment"],
        "source_result": result_path.as_posix(),
        "source_rows": rows_path.as_posix(),
        "status": "completed_development_audit",
        "row_count": row_count,
        "sessions_per_snr": len(seeds),
        "frames_per_session": frames,
        "arms": arms,
        "pairs": [list(pair) for pair in pairs],
        "windows_are_zero_based_half_open": True,
        "confidence_intervals": "未校正的跨 session 双侧 95% t 区间，仅作开发描述",
        "initial_state_timing_audit": timing_audit,
        "snrs": {},
    }
    report = [
        "# AR(1) 状态预测开发实验审计",
        "",
        f"实验：`{summary['experiment']}`；统计单位为独立 session，n={len(seeds)} / SNR。",
        f"已核验 {row_count} 条记录、{len(snrs)} 个 SNR、{len(seeds)} 个 seed、{frames} 帧和 {len(arms)} 臂；索引连续、各臂齐全、同帧波形哈希一致。",
        "窗口使用零起点左闭右开索引；首帧单独列为 `[0,1)`。每个窗口先在 session 内汇总错误数/比特数，再对 session 等权计算均值及未校正描述性 95% t 区间。",
        "",
        "## 窗口结果",
        "",
        "G 定义为配对对照的 BER(基线)−BER(候选)，正值表示候选 BER 较低。",
        "",
    ]
    for snr in snrs:
        snr_windows = []
        report += [f"### {snr} dB", "", "| 帧窗口 | " + " | ".join(arms) + " | " + " | ".join(
            f"G({base}−{candidate})" for base, candidate in pairs
        ) + " |", "|---|" + "---:|" * (len(arms) + len(pairs))]
        for start, end in windows:
            window_values = {arm: [] for arm in arms}
            pair_values = {pair: [] for pair in pairs}
            window_costs = {arm: [] for arm in arms}
            for seed in seeds:
                totals = {arm: [0, 0] for arm in arms}
                costs = {arm: [] for arm in arms}
                for index in range(start, end):
                    frame = grouped[(snr, seed, index)]
                    for arm in arms:
                        totals[arm][0] += frame[arm]["errors"]
                        totals[arm][1] += frame[arm]["bits"]
                        costs[arm].append(frame[arm]["elapsed_ms"])
                rates = {arm: totals[arm][0] / totals[arm][1] for arm in arms}
                for arm in arms:
                    window_values[arm].append(rates[arm])
                    window_costs[arm].append(sum(costs[arm]) / len(costs[arm]))
                for pair in pairs:
                    pair_values[pair].append(rates[pair[0]] - rates[pair[1]])
            arm_ci = {arm: _mean_ci(window_values[arm]) for arm in arms}
            pair_ci = {pair: _mean_ci(pair_values[pair]) for pair in pairs}
            costs_ci = {arm: _mean_ci(window_costs[arm]) for arm in arms}
            snr_windows.append({
                "start": start,
                "end_exclusive": end,
                "arm_mean_ci95_t": arm_ci,
                "pair_gain_mean_ci95_t": {"|".join(pair): value for pair, value in pair_ci.items()},
                "session_values": {
                    "arms": {arm: window_values[arm] for arm in arms},
                    "gains": {"|".join(pair): pair_values[pair] for pair in pairs},
                    "mean_elapsed_ms": window_costs,
                },
                "mean_elapsed_ms_ci95_t": costs_ci,
            })
            cells = [f"{arm_ci[arm][0]:.6g} [{arm_ci[arm][1]:.6g}, {arm_ci[arm][2]:.6g}]" for arm in arms]
            cells.extend(f"{pair_ci[pair][0]:+.6g} [{pair_ci[pair][1]:+.6g}, {pair_ci[pair][2]:+.6g}]" for pair in pairs)
            report.append(f"| {start}–{end} | " + " | ".join(cells) + " |")
        snr_summary = {"windows": snr_windows}
        summary["snrs"][str(snr)] = snr_summary
        report.append("")

        early_start, early_end = 0, min(16, frames)
        tail_start, tail_end = min(8192, frames - 1), frames
        changes = {arm: [] for arm in arms}
        gain_changes = {pair: [] for pair in pairs}
        for seed in seeds:
            rates = {}
            for start, end, label in ((early_start, early_end, "early"), (tail_start, tail_end, "tail")):
                rates[label] = {}
                for arm in arms:
                    errors = bits = 0
                    for index in range(start, end):
                        row = grouped[(snr, seed, index)][arm]
                        errors += row["errors"]
                        bits += row["bits"]
                    rates[label][arm] = errors / bits
            for arm in arms:
                changes[arm].append(rates["tail"][arm] - rates["early"][arm])
            for pair in pairs:
                gain_changes[pair].append(
                    (rates["tail"][pair[0]] - rates["tail"][pair[1]])
                    - (rates["early"][pair[0]] - rates["early"][pair[1]])
                )
        snr_summary["early_to_tail_change"] = {
            "early_window": [early_start, early_end],
            "tail_window": [tail_start, tail_end],
            "arm_delta_tail_minus_early_ci95_t": {arm: _mean_ci(values) for arm, values in changes.items()},
            "gain_delta_tail_minus_early_ci95_t": {"|".join(pair): _mean_ci(values) for pair, values in gain_changes.items()},
            "session_direction_counts": {
                "arm_ber_decrease": {arm: sum(value < 0 for value in values) for arm, values in changes.items()},
                "gain_increased": {"|".join(pair): sum(value > 0 for value in values) for pair, values in gain_changes.items()},
            },
        }
        report.append("末窗减首 16 帧的均值变化及 95% t 区间见汇总 JSON 的 `early_to_tail_change`；这是描述性开发比较，不单独证明自身改善、非下降或饱和。")
        report.append("")

    report += [
        "## 证据限制",
        "",
    ]
    if timing_audit["status"] == "known_initial_prediction_omission":
        report += [
            "记录的接收器源码指纹对应已审计的初始化遗漏：首个 Data Pilot 前没有状态预测，"
            "第二个 Data 帧才开始预测。当前 P6 的生成器在发送 acquisition 后已演化一次 CIR；"
            "因此不能将本实现描述为 acquisition 至首帧均已匹配的状态空间滤波。"
            "两条递归臂都有此遗漏，不代表配对影响抵消；以首帧为参照的自身改善和增益增长尤其需要保留这一限制。"
            "参见[首帧状态预测审计](../../continual_v2_initial_prediction_audit_20261120.md)。",
            "",
        ]
    report += [
        "本实验用于开发筛查，不构成独立确认。逐帧文件没有保存接收器后验协方差迹，也没有实现合法观测残差字段；因此本报告只评价 BER、配对增益和耗时，不能对协方差校准或残差匹配作结论。未预设并验证饱和等价容差，不能把末窗差异不显著解释成饱和。",
        "",
    ]
    return summary, "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    summary, report = summarize(args.result, args.manifest)
    summary_path = args.result.with_name(args.result.stem + "_summary.json")
    report_path = args.result.with_name(args.result.stem + "_report.md")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(f"已写入 {report_path} 与 {summary_path}")


if __name__ == "__main__":
    main()
