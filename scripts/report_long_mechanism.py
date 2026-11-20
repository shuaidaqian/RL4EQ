# -*- coding: utf-8 -*-
"""完整审计后报告预登记窗口的 BER、配对增量与物理/同步诊断。"""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.long_diagnostic_summary import paired_window_summary
from scripts.audit_long_diagnostics import audit, sha


def render_report(summary, manifest, audit_result):
    arms = [variant['name'] for variant in manifest['variants']]
    short_names = {variant['name']: variant['name'] for variant in manifest['variants']}
    aged_controls = [variant for variant in manifest['variants']
                     if variant['arm'] == 'current_pilot'
                     and variant.get('current_pilot_prior', 'acquisition') == 'aged_acquisition']
    static_controls = [variant for variant in manifest['variants']
                       if variant['arm'] == 'current_pilot'
                       and variant.get('current_pilot_prior', 'acquisition') == 'acquisition']
    online = 'recursive_pilot_ar1'
    if online not in arms:
        raise ValueError('本报告的机制诊断需要显式 recursive_pilot_ar1 臂。')
    aggregate = {(row['snr_db'], row['arm'], row['scope'], tuple(row['window'])): row
                 for row in summary['aggregate_windows']}
    windows = manifest['pre_registered_windows']
    n_sessions = len(manifest['seeds'])
    lines = [f"# {manifest['frames']} 帧开发会话：状态、同步与物理难度", '',
        f"全部 {audit_result['sessions']} 个 SNR-session 组合、{audit_result['rows']} 行及 "
        f"{audit_result['paired_frames']} 个配对帧审计通过。每会话 {manifest['frames']} 帧，"
        f"每个 SNR 有 {n_sessions} 个独立开发会话，"
        '同一 seed 在不同 SNR 的结果不额外计作独立会话。', '',
        '所有表格先在每个会话内按预登记窗口汇总，再对会话等权。报告从首个普通帧开始，'
        '不选择窗口、不拟合单调曲线。尚未冻结实质改善阈值、饱和等价容差或确认功效；'
        '本报告不给总体通过、等价或饱和判定。', '',
        '在线使用合法 acquisition 残差估噪；真值仅用于接收结束后的评分。神经权重冻结，'
        '当前物理解码分支不证明 Transformer 或离线闭环训练贡献。', '']
    if static_controls:
        lines += ['未老化 acquisition 先验的当前 Pilot 对照为：'
                  + '、'.join(variant['name'] for variant in static_controls)
                  + '。它与递归臂的差值混合历史观测及先验模型差异，不能全部归因于新增普通帧 Pilot。', '']
    if aged_controls:
        lines += ['显式使用老化 acquisition 先验的当前 Pilot 对照为：'
                  + '、'.join(variant['name'] for variant in aged_controls)
                  + '。请核对配套协议中冻结动态是否与递归臂相同；各臂自己的同步和 soft-tail '
                  '仍可能随解码轨迹变化，配对差描述整个递归接收策略的增量。', '']

    for snr in manifest['snrs']:
        lines += [f'## {snr} dB', '', '全会话结果：', '',
                  '| 臂 | 会话平均 BER | 总错误数 / 总比特数 |', '|---|---:|---:|']
        for arm in arms:
            row = aggregate[(snr, arm, 'whole_session', (0, manifest['frames']))]
            lines.append(f"| {short_names[arm]} | {row['mean_ber']:.8g} | {row['total_errors']} / {row['total_bits']} |")
        headers = ['窗口'] + [short_names[arm] + ' BER' for arm in arms] + [
            short_names[baseline] + '−' + short_names[online_arm] for baseline, online_arm in manifest['pairs']]
        lines += ['', '时间轨迹与配对增量（正增量表示在线臂 BER 较低）：', '',
                  '| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |']
        for window in windows:
            current = {arm: aggregate[(snr, arm, 'registered_window', tuple(window))] for arm in arms}
            values = [f'[{window[0]},{window[1]})'] + [f"{current[arm]['mean_ber']:.7g}" for arm in arms]
            values += [f"{current[baseline]['mean_ber'] - current[online_arm]['mean_ber']:+.7g}"
                       for baseline, online_arm in manifest['pairs']]
            lines.append('| ' + ' | '.join(values) + ' |')
        lines += ['', 'AR(1) 状态及同步诊断：', '',
            '| 窗口 | 仅相位对齐 NMSE | CIR 平方误差 | 先验协方差迹 | 逐帧残差/预测能量比均值 | Data 相位 RMS 的帧均值（rad） |',
            '|---|---:|---:|---:|---:|---:|']
        fields = ['phase_only_nmse_diagnostic', 'cir_phase_only_squared_error_diagnostic',
                  'prior_covariance_trace', 'pilot_innovation_energy_ratio_diagnostic',
                  'decoder_phase_data_rms_rad_diagnostic']
        for window in windows:
            means = aggregate[(snr, online, 'registered_window', tuple(window))]['means']
            lines.append('| ' + ' | '.join([f'[{window[0]},{window[1]})'] + [f'{means[field]:.7g}' for field in fields]) + ' |')
        lines += ['', '同波形的物理条件：', '',
            '| 窗口 | 真实 CIR 功率 | 频谱平坦度 | 全能量宽松额外信息 BER 下界 |', '|---|---:|---:|---:|']
        fields = ['true_cir_power_diagnostic', 'true_frequency_flatness_diagnostic', 'full_energy_loose_genie_bound_diagnostic']
        for window in windows:
            means = aggregate[(snr, online, 'registered_window', tuple(window))]['means']
            lines.append('| ' + ' | '.join([f'[{window[0]},{window[1]})'] + [f'{means[field]:.7g}' for field in fields]) + ' |')
        lines.append('')

    lines += ['## 解释限制', '',
        '- 逐会话配对差与所有相邻窗口的 ΔOnline、Δ基线、Δ增益保存在配套 JSON；Δ增益=Δ基线−ΔOnline。优势增长不等同于自身改善。',
        '- Pilot 相位由同批观测拟合，协方差未包含联合同步不确定性，能量比不能解释为已校准卡方统计。',
        '- CIR 平方误差与能量比先逐帧计算再平均；相位 RMS 为逐帧 RMS 的算术平均。包裹相位均值不跨会话算术平均。',
        '- 全能量下界先逐帧计算再平均，使用事后真实噪声与功率，未计帧尾截断；不是合法接收 BER，也不是精确帧内边界。',
        '- 状态、同步、物理条件与 BER 的伴随变化不构成单因素因果证明；不能用这份开发报告宣称目标不可达。',
        '- 接收步骤 elapsed_ms 不含事后评分和落盘；本实验不作端到端实时验收。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', required=True)
    parser.add_argument('--source-snapshot', help='在当前源码已升级时，核验归档的完整运行源码与输入字节。')
    args = parser.parse_args()
    result_path = Path(args.result)
    result = json.loads(result_path.read_text(encoding='utf8'))
    if result.get('status') != 'completed':
        raise ValueError('所有预定会话完成前禁止生成性能报告。')
    audit_result = audit(result_path, source_snapshot_path=args.source_snapshot)
    manifest = audit_result['manifest']
    summary = paired_window_summary(audit_result, manifest)
    summary['result_sha256'] = audit_result['result_sha256']
    summary['rows_sha256'] = audit_result['rows_sha256']
    summary['manifest_sha256'] = audit_result['manifest_sha256']
    summary['source_verification'] = audit_result['source_verification']
    summary['source_snapshot_sha256'] = audit_result['source_snapshot_sha256']
    summary['analysis_git_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    summary['analysis_sources'] = {path: sha(ROOT / path) for path in (
        'scripts/report_long_mechanism.py', 'scripts/audit_long_diagnostics.py',
        'evaluation/long_diagnostic_summary.py', 'evaluation/source_snapshot.py')}
    text = render_report(summary, manifest, audit_result)
    result_path.with_name(result_path.stem + '_audit.json').write_text(
        json.dumps(audit_result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    result_path.with_name(result_path.stem + '_paired_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    result_path.with_name(result_path.stem + '_report.md').write_text(text, encoding='utf8')
    print('完整审计及全部预登记窗口报告已生成；未运行研究会话，不作独立确认。')


if __name__ == '__main__':
    main()
