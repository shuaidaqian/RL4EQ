# 第二十三轮：Pilot上下文阻尼Gauss–Newton数值审计

## 固定审计条件

本轮只验证在线更新器的合法性、数值稳定性和既有开发Data行为。使用v22相同的2维 `pilot_context_mlp_v1` 表示、四个SNR、三个开发seed、接收缓存和A/B checkpoint；信道、Pilot、接收窗口与标签边界均不变。候选仅由当前帧Adapt Pilot BCE及其Jacobian/Fisher曲率产生，验证Pilot仅在候选形成后作接受/拒绝与离线锚点回退，Data只用于门控结束后的事后评分。

固定参数：阻尼`0.01`、近端权重`0.1`、Adapt温度`2.0`、逐步半径`0.15`、总半径`0.8`、最多8次Adapt目标回溯；所有SNR共用，不做按SNR调度。配置见[`v23审计清单`](../../../../configs/audits/detection_subspace_long_v23_gauss_newton_numerical_v1.json)。

## 结果

| SNR | A Frozen BER | A Online BER | A增益 | B Frozen BER | B Online BER | B增益（95%区间） | B−A交互 | Jacobian满秩帧比例 | 平均Fisher条件数 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.15894912 | 0.15897624 | -2.71e-05 | 0.15894912 | 0.15897624 | -2.71e-05 [-8.55e-05, +3.12e-05] | +0 | 100% | 1.0058 |
| 5 dB | 0.026448568 | 0.026448568 | +0 | 0.026448568 | 0.026462132 | -1.36e-05 [-7.19e-05, +4.48e-05] | -1.36e-05 | 100% | 1.0201 |
| 10 dB | 0.0018581815 | 0.0018581815 | +0 | 0.0018581815 | 0.0018581815 | +0 [+0, +0] | +0 | 100% | 1.0070 |
| 15 dB | 8.1380211e-05 | 8.1380211e-05 | +0 | 8.1380211e-05 | 8.1380211e-05 | +0 [+0, +0] | +0 | 100% | 1.0002 |

四个SNR上Jacobian均有限且满2秩，阻尼Fisher矩阵条件数接近1，数值稳定性良好。但0 dB的A、B臂都比Frozen退化约`2.71e-5`；5 dB A臂不变、B臂退化约`1.36e-5`；10/15 dB不变。所有B−A差异均未形成正增益。验证Pilot守门虽通过了不少候选，仍未带来Data增益。这既未证明在线优于Frozen，也未证明v22的离线学习更新器在新优化器下有额外贡献。

## 结论与下一步

阻尼Gauss–Newton在2维合法Pilot信息上可稳定求解，但直接套用到v22已训练表示并未产生在线BER增益。验证Pilot优化方向与Data收益之间仍存在偏差。下一轮应在相同Adapt损失、验证守门、持久化、半径和时序下，匹配实现可微Gauss–Newton内循环与普通监督/元训练对照，检查离线表示能否学到与该更新算子匹配的初始化/局部几何；训练seed与选择/评估seed保持隔离。只有开发证据出现可归因的在线正增益才继续扩展，不启动正式确认。

## 复现与边界

- 运行命令：`python scripts/audit_context_gauss_newton.py --feature-cache logs/detection_subspace_training_v22_history1_crossseed_v1_20261105/evaluation_features.pt --checkpoint-a logs/detection_subspace_training_v22_history1_crossseed_v1_20261105/A_supervised.pt --checkpoint-b logs/detection_subspace_training_v22_history1_crossseed_v1_20261105/B_post_adapt.pt --output-dir <新输出目录> --device cuda`。
- A/B合计768条帧记录（每臂384条）；输入文件、配置、源码和压缩逐帧指标指纹见[`summary.json`](summary.json)，逐帧结果见[`frame_metrics.jsonl.gz`](frame_metrics.jsonl.gz)。
- `matched_offline_training=false`、`core_confirmed=false`、`confirmation_ready=false`。该审计是开发机制诊断，不能作为独立确认。
