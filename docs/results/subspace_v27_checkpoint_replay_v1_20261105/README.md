# v27冻结checkpoint独立开发会话复核

## 目的与控制

本复核检验v27新开发seed中0/5 dB的正向Online−Frozen点估计能否迁移到未参与训练、选择或先前评估的新会话。A/B checkpoint固定为v27最终选择权重，不重新训练、不重新选择；只生成新信道会话并按原在线更新、验证Pilot守门、回退规则执行。seed固定为`121800/121801/121802`，每seed/SNR 32帧，完整报告0/5/10/15 dB。信道、PDP、同步/漂移、Pilot、接收窗口和信息边界未改。配置见[`config.json`](config.json)，执行入口为[`replay_subspace_checkpoints.py`](../../../../scripts/replay_subspace_checkpoints.py)。

## 结果

增益为 `BER_frozen - BER_online`，正值表示在线更好。

| SNR | A增益（95%区间） | B增益（95%区间） | B−A交互 | 结论 |
|---:|---:|---:|---:|---|
| 0 dB | +6.78e-05 [-1.43e-04, +2.78e-04] | +1.36e-05 [-4.48e-05, +7.19e-05] | -5.43e-05 | B正点估计小且低于A |
| 5 dB | -5.43e-05 [-1.13e-04, +4.11e-06] | -8.14e-05 [-2.84e-04, +1.21e-04] | -2.71e-05 | A/B均退化，B更差 |
| 10 dB | -1.36e-05 [-7.19e-05, +4.48e-05] | -1.36e-05 [-7.19e-05, +4.48e-05] | 0 | A/B点估计相同且退化 |
| 15 dB | 0 | 0 | 0 | 没有接受更新，BER持平 |

v27在0/5 dB的B臂正向点估计未能复现：0 dB的B−A交互转负，5 dB在线退化。10 dB也出现同样的小幅退化，15 dB仍无更新。所有可估95%区间跨零；三个开发seed只构成机制复核，不构成正式确认。逐帧记录和来源/权重指纹见[`summary.json`](summary.json)及[`frame_metrics.jsonl.gz`](frame_metrics.jsonl.gz)。A/B checkpoint一并保存在[v27训练归档](../detection_subspace_long_v27_gauss_newton_online_gain_proximal001_v1_20261105/)。

## 结论与后续

停止继续扩大当前`pilot_context_mlp_v1 + 单帧阻尼Gauss–Newton + prox=0.01 + online_gain`组合。失败不是Pilot Jacobian秩不足：v27四SNR几何均满秩；关键困难是合法Pilot方向对Data均衡目标不稳定，且独立会话未复现B相对普通监督A的收益。

下一步回到R2的更新表示设计：先用已有离线开发缓存做只读的Pilot可辨识性分析，检查由Adapt Pilot可观测残差构造的有限维更新方向，能否在未参与拟合的开发会话上预测更新后的Data改善；Data标签只作离线事后评分，不参与在线更新/门控。通过该筛查后再设计新的Pilot条件更新器与匹配离线训练；若没有可辨识、跨会话一致的方向，则先修改离线表示/在线可观测目标，不再运行同一GN配置的seed矩阵。正式确认不启动。

## 验证状态

- 768条开发复核记录，A/B各自3个新seed，四个SNR完整。
- 冻结checkpoint；不含训练与选择过程。
- 独立复核清单与入口定向测试`2 passed`；全量pytest`959 passed in 336.13s`。
- `core_confirmed=false`、`confirmation_ready=false`。
