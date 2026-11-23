# -*- coding: utf-8 -*-
"""把 P5 确认分析 JSON 渲染为 Markdown 报告；不重新估计或挑选结果。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def fmt(value, digits=5):
    if value is None:
        return '—'
    if isinstance(value, float) and abs(value) < 0.01 and value != 0:
        return f'{value:.3e}'
    return f'{value:.{digits}f}'


def ci(values):
    if not values or values[0] is None:
        return '—'
    return f'[{values[0]:.5f}, {values[1]:.5f}]'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analysis = json.loads((ROOT / args.analysis).read_text(encoding='utf8'))
    audit = json.loads((ROOT / args.audit).read_text(encoding='utf8'))
    lines = ['# P5 四 SNR 独立确认报告', '',
             f'- 协议：`{analysis["protocol"]}`',
             f'- 完整性：{"通过" if audit.get("all_complete") else "未通过"}',
             f'- 统计单位：session；主族 Holm，次族/保持族分别 Holm。', '',
             '## 结果', '',
             '| SNR | n | 自身改善 (CI95) | 相对当前 Pilot 增量 (CI95) | Holm p | 相对 Frozen（描述） | 保持 TOST p (Holm) |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for snr in sorted(analysis['results'], key=int):
        result = analysis['results'][snr]
        own = result['own_gain']
        adv = result['adv_current']
        hold = result['tests']['hold_tost']
        lines.append(
            f"| {snr} | {result['session_count']} | {fmt(own['mean'])} {ci(own['ci95'])} | "
            f"{fmt(adv['mean'])} {ci(adv['ci95'])} | "
            f"{fmt(result['tests']['adv_current'].get('holm_pvalue'))} | "
            f"{fmt(result['adv_frozen']['mean'])} | "
            f"{fmt(hold['pvalue'])} ({fmt(hold.get('holm_pvalue'))}) |")
    lines += ['', '## 判读', '']
    primary_ok = all(
        analysis['results'][snr]['tests']['adv_current'].get('holm_pvalue', 1.0) < 0.05
        and analysis['results'][snr]['adv_current']['mean'] > 0
        for snr in analysis['results'])
    secondary_ok = all(
        analysis['results'][snr]['tests']['own_minimum_effect'].get('holm_pvalue', 1.0) < 0.05
        for snr in analysis['results'] if int(snr) in (0, 5))
    hold_ok = all(
        analysis['results'][snr]['tests']['hold_tost'].get('holm_pvalue', 1.0) < 0.05
        for snr in analysis['results'])
    lines.append(f'- 主族（四 SNR 相对当前 Pilot 增量）：{"确认" if primary_ok else "未确认"}。')
    lines.append(f'- 次族（0/5 dB 自身改善超过 0.005）：{"确认" if secondary_ok else "未确认"}。')
    lines.append(f'- 保持 TOST："{"确认" if hold_ok else "未确认"}。')
    lines.append('- 10/15 dB 自身改善只作描述性报告；真实 ESNR/轨迹诊断仅用于事后解释。')
    lines += ['', '## 限制', '',
              '- 确认集一次性运行，未使用可选停止；所有 session、SNR 与失败均需保留在归档中。',
              '- 相对 Frozen 的优势受 Frozen 后段退化放大，不能替代自身改善。',
              '- 若任一核心主张未满足，按协议报告失败，不在同一确认集上修改方法。']
    output = ROOT / args.output
    output.write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('wrote', output)


if __name__ == '__main__':
    main()
