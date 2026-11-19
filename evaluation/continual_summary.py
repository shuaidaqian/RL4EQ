# -*- coding: utf-8 -*-
"""会话级持续均衡诊断；只描述开发数据，不以斜率替代形状确认。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import t


PAIRS = (("acquisition", "recursive_pilot"), ("current_pilot", "recursive_pilot"),
         ("recursive_pilot", "recursive_soft"))


def interval(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("会话统计需要非空有限样本。")
    center = float(values.mean())
    deviation = float(values.std(ddof=1)) if len(values) > 1 else None
    margin = (float(t.ppf(.975, len(values) - 1) * deviation / len(values) ** .5)
              if deviation is not None and deviation > 0 else None)
    return {"mean": center, "ci95": None if margin is None else [center - margin, center + margin],
            "sessions": len(values), "interval_status": (
                "too_few_sessions" if len(values) == 1 else
                "zero_empirical_variance_not_equivalence" if deviation == 0 else "t_descriptive")}


def summarize_development(result):
    if result.get("status") != "completed":
        raise ValueError("实验未完成，不能生成完整研究结论。")
    groups = defaultdict(dict)
    for row in result["rows"]:
        key = (row["snr_db"], row["arm"], row["seed"])
        if row["frame_index"] in groups[key]:
            raise ValueError("重复帧记录。")
        groups[key][row["frame_index"]] = row
    expected = int(result["protocol"]["frames"])
    for frames in groups.values():
        if set(frames) != set(range(expected)):
            raise ValueError("会话缺帧或帧索引非法。")
    summary = {"status": "development_descriptive_only", "statistical_unit": "session", "per_snr": {}}

    def ber(frames, indices):
        selected = [frames[i] for i in indices]
        return sum(r["errors"] for r in selected) / sum(r["bits"] for r in selected)

    for snr in result["protocol"]["snrs"]:
        current = {"arms": {}, "paired": {}, "noise_ratio": None}
        seeds = result["protocol"]["seeds"]
        edges = sorted(set(x for x in [0, 1, 4, 8, 16, 32, 64, 128, expected] if x <= expected))
        for arm in result["protocol"]["arms"]:
            frames = [groups[(snr, arm, seed)] for seed in seeds]
            window_values = [[ber(f, range(a, b)) for f in frames] for a, b in zip(edges, edges[1:])]
            late = range(max(0, expected - max(1, expected // 4)), expected)
            first = range(min(4, expected))
            current["arms"][arm] = {
                "ber": interval([ber(f, range(expected)) for f in frames]),
                "early_minus_late_ber": interval([ber(f, first) - ber(f, late) for f in frames]),
                "windows": [{"start": a, "stop": b, **interval(values)}
                            for a, b, values in zip(edges, edges[1:], window_values)],
                "mean_elapsed_ms": interval([np.mean([r["elapsed_ms"] for r in f.values()]) for f in frames]),
            }
        for baseline, online in result["protocol"].get("pairs", PAIRS):
            if baseline not in result["protocol"]["arms"] or online not in result["protocol"]["arms"]:
                continue
            values, late_growth = [], []
            for seed in seeds:
                b, o = groups[(snr, baseline, seed)], groups[(snr, online, seed)]
                if any(b[i]["rx_sha256"] != o[i]["rx_sha256"] for i in b):
                    raise ValueError("接收臂波形不匹配。")
                values.append(ber(b, range(expected)) - ber(o, range(expected)))
                late_growth.append((ber(b, late) - ber(o, late)) - (ber(b, first) - ber(o, first)))
            current["paired"][f"{baseline}_minus_{online}"] = {
                "gain": interval(values), "positive_sessions": sum(x > 0 for x in values),
                "late_minus_early_gain": interval(late_growth),
            }
        noise = [x["noise_estimate"] / x["noise_truth_diagnostic"] for x in result["acquisitions"]
                 if x["snr_db"] == snr]
        current["noise_ratio"] = interval(noise)
        summary["per_snr"][str(snr)] = current
    return summary


def markdown_report(result, summary):
    if "variants" in result["protocol"]:
        return variants_report(result, summary)
    lines = ["# 持续均衡 v2：四臂开发筛查", "",
             "状态：**开发诊断，不是独立确认，不证明增长至饱和。**", "",
             f"实验：`{result['protocol']['experiment']}`；{len(result['protocol']['seeds'])} 个会话 × "
             f"{result['protocol']['frames']} 帧；每臂每帧一次神经前向。", "",
             "噪声来自 acquisition 残差估计；所有臂同波形、同初始化、同 checkpoint。", "",
             "| SNR | acquisition | 当前 Pilot | 递归 Pilot | 递归 soft | 估噪/真值 |",
             "|---|---:|---:|---:|---:|---:|"]
    for snr, value in summary["per_snr"].items():
        scores = [value["arms"].get(arm, {}).get("ber", {}).get("mean")
                  for arm in ["acquisition", "current_pilot", "recursive_pilot", "recursive_soft"]]
        formatted = ["—" if x is None else f"{x:.6f}" for x in scores]
        lines.append(f"| {snr} | " + " | ".join(formatted) + f" | {value['noise_ratio']['mean']:.3f} |")
    lines += ["", "## 会话级配对差", "",
              "正值表示后一方法 BER 更低。区间为未作多重性校正的双侧 t 区间，仅作开发诊断。", "",
              "| SNR | 对比（前减后） | 均值 | 95% 区间 | 正向会话 |",
              "|---|---|---:|---|---:|"]
    for snr, value in summary["per_snr"].items():
        for pair, comparison in value["paired"].items():
            gain = comparison["gain"]
            ci = gain["ci95"]
            text = "样本不足或方差退化" if ci is None else f"[{ci[0]:+.6f}, {ci[1]:+.6f}]"
            lines.append(f"| {snr} | {pair} | {gain['mean']:+.6f} | {text} | "
                         f"{comparison['positive_sessions']}/{gain['sessions']} |")
    lines += ["", "## 边界", "",
              "- 当前是既有 checkpoint 的接收机制筛查，尚未进行闭环重训。",
              "- 噪声残差可能包含同步/信道模型误差，估计比值只作事后诊断。",
              "- 首帧已经使用当前 Pilot，首帧优势不能作为跨帧增长证据。",
              "- 自身 BER 与相对优势分别检查；短时开发曲线不能证明饱和。",
              "- 各窗口、逐会话差值与耗时见同目录统计 JSON；无选择性删 seed。", ""]
    return "\n".join(lines)


def variants_report(result, summary):
    """候选可能改变同步、checkpoint 或分支，逐项明示后作开发比较。"""
    protocol = result['protocol']
    lines = ['# 持续均衡候选配对开发比较', '',
             '**开发诊断，不是独立确认，不证明增长至饱和。**', '',
             f"实验：`{protocol['experiment']}`；{len(protocol['seeds'])} 会话 × {protocol['frames']} 帧。", '',
             '所有候选同波形、同 acquisition，噪声由 acquisition 合法估计。', '',
             '| 候选 | 接收状态/上下文来源 | 相位估计 | 解码分支 | checkpoint |',
             '|---|---|---|---|---|']
    for row in protocol['variants']:
        state = f"共享 {row['context_source']} 的实际输入" if row.get('context_source') else row['arm']
        lines.append(f"| {row['name']} | {state} | {row.get('phase_estimator', 'unwrap_median')} | "
                     f"{row.get('decoder', 'neural')} | `{row['checkpoint']}` |")
    lines += ['', '| SNR | 候选 | 平均 BER | 早期减晚期 BER | 平均 ms/帧 |',
              '|---|---|---:|---:|---:|']
    for snr, value in summary['per_snr'].items():
        for name, row in value['arms'].items():
            lines.append(f"| {snr} | {name} | {row['ber']['mean']:.6f} | "
                         f"{row['early_minus_late_ber']['mean']:+.6f} | {row['mean_elapsed_ms']['mean']:.2f} |")
    lines += ['', '配对差为前者 BER 减后者 BER；正值表示后者更好。双侧 t 区间未作多重性校正。', '',
              '| SNR | 比较 | 均值 | 95% 区间 | 正向会话 |', '|---|---|---:|---|---:|']
    for snr, value in summary['per_snr'].items():
        for pair, row in value['paired'].items():
            stat = row['gain']
            ci = '样本不足或方差退化' if stat['ci95'] is None else f"[{stat['ci95'][0]:+.6f}, {stat['ci95'][1]:+.6f}]"
            lines.append(f"| {snr} | {pair} | {stat['mean']:+.6f} | {ci} | {row['positive_sessions']}/{stat['sessions']} |")
    lines += ['', '物理分支仍保留同一 checkpoint 的已学习同步，不是完全非神经基线。',
              '各窗口、自身改善、优势增长与成本分别报告；短期均值改善不能替代增长与饱和确认。', '']
    return '\n'.join(lines)
