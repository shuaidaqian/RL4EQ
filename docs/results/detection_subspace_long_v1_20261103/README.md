import gzip, hashlib, json
from pathlib import Path
src=Path('logs/detection_subspace_training_v1_20261103/frame_metrics.jsonl')
dst=Path('docs/results/detection_subspace_long_v1_20261103/frame_metrics.jsonl.gz')
with src.open('rb') as f, gzip.GzipFile(filename=str(dst), mode='wb', mtime=0) as out:
    out.write(f.read())
summary=json.loads(Path('logs/detection_subspace_training_v1_20261103/summary.json').read_text(encoding='utf-8'))
inspection=json.loads(Path('logs/detection_subspace_inspection_v1_20261103/inspection.json').read_text(encoding='utf-8'))
# 只保留可复核的简洁展开；完整几何和稠密轨迹仍在 inspection.json/原始gzip中。
analysis={'schema_version':'detection-subspace-development-analysis-v1','role':'development','source_summary_sha256':hashlib.sha256(Path('logs/detection_subspace_training_v1_20261103/summary.json').read_bytes()).hexdigest(),'source_inspection_sha256':hashlib.sha256(Path('logs/detection_subspace_inspection_v1_20261103/inspection.json').read_bytes()).hexdigest(),'dimension':summary['dimension'],'records':summary['records'],'per_snr':summary['per_snr'],'geometry':{'eligible_dimensions':inspection['eligible_dimensions'],'selected_dimension':inspection['selected_dimension'],'reachability_smoke':inspection['reachability_smoke']},'dense_memory_diagnostic':{'rows':len(inspection['dense_memory_rows']),'per_snr':[]},'conclusion':'结构4维在固定长回波上可辨识且有作用，但A/B四臂开发均未显示可信在线Data增益；核心目标未完成。','confirmation_ready':False}
for snr in [0,5,10,15]:
    rows=[r for r in inspection['dense_memory_rows'] if r['snr_db']==snr]
    analysis['dense_memory_diagnostic']['per_snr'].append({'snr_db':snr,'accepted':sum(r['accepted'] for r in rows),'harmful_new_accept':sum(r['accepted'] and r['ber_retained']>r['ber_before'] for r in rows),'harmful_old_memory':sum(r['ber_before']>r['ber_P'] for r in rows),'mean_ber_P':sum(r['ber_P'] for r in rows)/len(rows),'mean_ber_before':sum(r['ber_before'] for r in rows)/len(rows),'mean_ber_candidate':sum(r['ber_candidate'] for r in rows)/len(rows),'mean_ber_retained':sum(r['ber_retained'] for r in rows)/len(rows)})
Path('docs/results/detection_subspace_long_v1_20261103/analysis.json').write_text(json.dumps(analysis,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
PY
& 'D:\Research\RL4EQ\.venv-gpu\Scripts\python.exe' -
@'
# R2 固定长回波检测子空间开发归档

这是一批固定现有信道配置的 R2 开发数据，不是 R3 正式确认。信道配置仍为 `eme_long_memory_v2`：1024 符号帧、116 符号记忆、Adapt224/验证32、116 样本后缀、一次 acquisition；清单和 checkpoint SHA256 见 `manifest.json` 与 `summary.json`。

R1 已知的稠密234实数记忆在本批64帧补充诊断中被拆成两类问题：验证 Pilot 通过后的新候选仍可能提高 Data BER；而部分帧在更新前已有旧记忆相对 P 的退化。因而不能只调验证阈值或直接取消累计保护。

固定门槛在帧0/7、四个 SNR 和两个检查 seed 上筛选 2/4/8 个在线实数系数：2维、4维满列秩且条件数均低于100，8维最差条件数约403被淘汰。按预注册规则选择4维。普通监督8步作用检查使基网络、系数和 Data 输出都实际改变；这只证明可训练，不证明泛化。

A 为普通监督，B 为匹配在线更新目标；各自64次外层更新，从相同初始化和训练会话开始，在独立选择会话上固定第64步 checkpoint。随后在3个评估 seed、四SNR、每32帧的开发会话上生成768条 A/B 同波形记录。四臂的在线控制不读取 Data 标签；Data 仅在动作完成后由离线评分读取。

结果在 `analysis.json` 和 `summary.json` 中。A/B 的 Online 相对同源 Frozen BER 区间均跨零，B 没有稳定优于 A 的交互增量；15 dB 错误过少，不能解释为非劣。验证 Pilot 仍出现少量 Data 有害保留，当前结构与守门任务尚未完成可迁移证明。

`frame_metrics.jsonl.gz` 是完整768条原始轨迹的 gzip 归档，mtime固定为0；解压字节 SHA256记录于以下校验命令输出中。缓存特征（未提交）和来源源码指纹见 `summary.json`，提交只保留可复核的指标、清单和完整压缩轨迹。

本阶段结论：**R2 结构可辨识且有作用，但固定长回波开发集尚未证明离线训练适应信道能带来可靠在线均衡增益，核心目标仍未完成。** 下一步应围绕更新目标、验证留出集对 Data 的代表性、历史参数有效期和受控重置/遗忘设计做单因素机制修正；不扩大当前 seed 矩阵，不进入 R3/R4/R5/R6。

验证：`tests/test_detection_subspace.py` 等定向测试26项通过；完整 pytest **903 passed，415.52秒**。GPU几何审计约116.20秒；四臂训练与评估约525.43秒。研究计时不等于部署端到端延迟。
