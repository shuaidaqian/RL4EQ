# 强离线模型 head-Adapt Fisher 几何诊断 v1

## 目的与协议

本诊断判断强离线均衡器的共享输出头 `head`（65个实参数）在现有合法 Adapt Pilot 下能支持多少有效更新方向，以及主信息方向能否跨会话、跨 SNR 保持一致。它不执行在线更新，不读 Pilot/Data 发送标签，也不计算 Data 指标；结果只用于确定低秩更新器候选维数。

来源模型是 `eme_bce_all_32_20260905_pilot256`，SHA256=`8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。沿用 Level B、116符号长回波、前缀 Pilot=256（Adapt=224、Reward=32）、0/5/10/15 dB，使用3个此前未用于这项诊断的会话 `62000–62002`。每个 session/SNR 仅分析首帧 Adapt。接收波形与标签范围由 `visible="adapt"` 固定。对展开均衡器完整前向求最终 Adapt logits 对 65 个 `head` 参数的精确 Jacobian，再以 logistic Fisher `Jᵀ diag(p(1-p)) J / N` 汇总；因此包含共享 head 多次展开后对最终输出的间接影响。

数值秩阈值为最大特征值的 `1e-6`。谱有效秩定义为特征值归一化后的熵指数。主方向相似度取特征向量余弦绝对值，消除符号不定性。

## 结果

| SNR | 三会话数值秩范围 | 谱有效秩范围 | 最大特征方向能量占比 | 前两方向累计能量 | 会话间主方向余弦范围 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 15–18 | 1.291–1.319 | 94.52%–94.84% | 97.05%–98.07% | 0.919–0.982 |
| 5 dB | 12–13 | 1.215–1.226 | 95.92%–96.22% | 98.83%–99.11% | 0.981–0.995 |
| 10 dB | 7–10 | 1.191–1.217 | 95.63%–96.45% | 98.89%–99.59% | 0.971–0.991 |
| 15 dB | 2–4 | 1.000–1.163 | 96.54%–100.00% | 99.99%–100.00% | 0.913–0.990 |

按同一 session 配对跨 SNR，主方向平均余弦在 0/5、0/10、0/15、5/10、5/15、10/15 dB 间分别为 `0.973、0.968、0.935、0.995、0.972、0.980`。因此当前强离线 head 的 Adapt Fisher 虽有多维数值秩，但绝大部分局部可辨信息集中在一个跨会话、跨 SNR 高度相似的方向。高 SNR 的其余有效秩更低，符合 Pilot 梯度信息随误差减小而下降的预期。

这是一个重要的结构证据：下一步可把在线可训练状态压至一个共享的 rank-1 head 神经子空间，测试减少 Adam 坐标噪声和交叉方向伤害是否提升 Reward 守门后的 Data 检测。它**不能**证明该方向改善 Data；Fisher 描述的是 Adapt 参数信息，不是 Adapt 到 Data 的梯度传输，更不构成在线增益或随帧增长证据。下一实验仍须直接比较同源 Frozen 与持续 rank-1 Online，并与一个固定 Fisher 方向参照配对；若开发集显示增益，再训练离线外层学习该共享方向，不使用本诊断会话作独立确认。

## 产物与复现

- 配置：[eme_long_memory_v2_update_direction_diagnostic_v1.json](../../../configs/eme_long_memory_v2_update_direction_diagnostic_v1.json)
- 脚本：[diagnose_head_fisher.py](../../../scripts/diagnose_head_fisher.py)
- 首轮谱值汇总：[summary.json](summary.json)；包含主特征方向与输入波形指纹的最终诊断：[summary_with_vectors.json](summary_with_vectors.json)
- JSON SHA256=`ca9c62188e7762070bcbe865386cdc82e0785cfc1767ad999ab1316e2a5c6715`；checkpoint SHA256 如上。
- 在线控制、Reward验收或Data反馈均未执行；开发 session 不用作后续独立确认。
