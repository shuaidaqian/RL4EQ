# 历史在线路线索引

状态：历史诊断/已被当前主路线取代。保留本索引和原始实验数据用于复现、理解失败机制；不再作为待执行方案。

当前唯一主路线见 [Pilot-only 状态跟踪接收机](state_tracking_receiver_design.md)；实验门槛见 [验证协议](state_tracking_validation_protocol.md)。

| 历史路线 | 代表文件 | 现有证据与处理 |
|---|---|---|
| Adapt Pilot BCE/PEFT 更新 + Reward BCE 守门 | `training/online_adaptation.py`、`docs/eme_guarded_online_state_recovery_results.md` | Reward proxy 与 Data BER 可负相关；PEFT 未稳定超过 F1；保留用于诊断，不作为默认入口 |
| Contextual Bandit | `agent/safe_contextual_bandit.py`、`docs/eme_online_rl_findings.md` | action catalog/代理排序尚未满足准入条件；不进入主比较 |
| 窗口离散 PPO 调制 | `training/windowed_discrete_ppo.py`、`docs/eme_online_rl_findings.md` | 窗口 reward 不等于未来 Data BER；动作与状态回滚还需完整原子性审计 |
| 连续调制 PPO / Continual PPO | `training/rl_modulated_online.py`、`training/continual_ppo.py` | 旧路线；不作为新主实验入口 |
| Pilot replay 排序探索 | `docs/pilot_replay_protocol_audit_20261014.md`、`docs/pilot_holdout_waveform_isolation_20261017.md` | 保留协议修正和排序失败结论；代理未通过独立轨迹门槛 |
| PEFT 微调探索 | `docs/online_peft_phase_trend_head_route.md`、`docs/online_logit_guard_replay_20261015.md` | 保留数值；局部 seed/SNR 改善不能证明稳定正相关或全配置收益 |

新增代码不得悄悄把这些路线恢复为默认。若作专项消融，输出必须标明 checkpoint、候选状态、完整 tail 回滚、Data 隔离和代理排序通过状态。
