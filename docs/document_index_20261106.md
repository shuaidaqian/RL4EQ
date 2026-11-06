# 全仓文档索引（2026-11-06）

本索引由 `scripts/build_document_index.py` 生成。它只读取路径，不改写历史文档。

- 当前规范：[research_route_v4_20261106.md](research_route_v4_20261106.md)。
- 旧文档保留原语义；被取代关系见下表，不按 v4 追认旧实验成功。
- 结果归档的数值与结论以各自 README 为准；本索引只标注状态。

## 1. 当前规范与关键入口

| 类别 | 文档 | 用途 |
|---|---|---|
| 当前规范 | [docs/research_route_v4_20261106.md](research_route_v4_20261106.md) | 唯一当前路线、自由度、方法与阶段 |
| 当前规范 | [docs/protocols/research_route_v4_20261106.json](protocols/research_route_v4_20261106.json) | 机器可读路线状态 |
| 当前规范 | [docs/protocols/confirmation_criteria_v4_draft_20261106.md](protocols/confirmation_criteria_v4_draft_20261106.md) | 确认判据草案（未冻结） |
| 当前规范 | [docs/document_index_20261106.md](document_index_20261106.md) | 本索引 |
| 进展总表 | [docs/online_adaptation_progress_20261106.md](online_adaptation_progress_20261106.md) | 在线微调阶段进展与失败路线 |
| 离线基线 | [docs/offline_equalizer_training_and_baseline_20261105.md](offline_equalizer_training_and_baseline_20261105.md) | 离线模型与传统基线主矩阵 |
| P0 结果 | [docs/results/tail_alignment_v1_20261106/README.md](results/tail_alignment_v1_20261106/README.md) | soft-tail 训练/评估对齐 |
| 统计功效 | [docs/results/power_error_composition_20261106/README.md](results/power_error_composition_20261106/README.md) | MDE、session 数与错误构成 |
| 方法调研 | [docs/results/method_survey_20261106/README.md](results/method_survey_20261106/README.md) | 在线适配机制与文献 |
| 离线饱和 | [docs/results/offline_saturation_20261106/README.md](results/offline_saturation_20261106/README.md) | 训练预算/宽度接近饱和；CIR 是 5/10 dB 主要残差 |
| P1 诊断 | [docs/results/data_aided_elbo_diag_v1_20261106/README.md](results/data_aided_elbo_diag_v1_20261106/README.md) | Data 软边缘化 ELBO 方向筛查 |
| P1 诊断 | [docs/results/data_aided_cir_diag_v1_20261106/README.md](results/data_aided_cir_diag_v1_20261106/README.md) | Data-aided CIR 四模式对照（负/中性） |
| P1 结果 | [docs/results/bayesian_cir_diag_v1_20261106/README.md](results/bayesian_cir_diag_v1_20261106/README.md) | 贝叶斯递归 CIR：四 SNR 同向物理增益 |
| P1 结果 | [docs/results/bayes_head_diag_v1_20261106/README.md](results/bayes_head_diag_v1_20261106/README.md) | 共享 Bayes CIR 后的 head 增量（负） |
| 近期机制 | [docs/results/rank1_state_filter_dev_v1_20261106/README.md](results/rank1_state_filter_dev_v1_20261106/README.md) | rank-1 状态滤波开发筛查 |
| 近期机制 | [docs/results/full_head_softber_train32_v1_20261106/README.md](results/full_head_softber_train32_v1_20261106/README.md) | soft-BER 全 head trajectory |
| 近期机制 | [docs/results/subspace_v27_checkpoint_replay_v1_20261105/README.md](results/subspace_v27_checkpoint_replay_v1_20261105/README.md) | v27 冻结 checkpoint 复核 |

## 2. 取代关系

| 旧文档/旧协议 | 当前替代 | 说明 |
|---|---|---|
| docs/research_route_v3_20261103.md | docs/research_route_v4_20261106.md | 当前路线 |
| docs/protocols/research_route_v3_20261103.json | docs/protocols/research_route_v4_20261106.json | 机器可读状态 |
| docs/protocols/confirmation_criteria_draft_20261105.md | docs/protocols/confirmation_criteria_v4_draft_20261106.md | 确认判据草案 |
| docs/results/**/README.md（v14–v27） | docs/research_route_v4_20261106.md | 保留原语义；不按 v4 追认成功 |
| pilot_only tail 协议 | configs/eme_long_memory_v2_online_dd_tail_v1.json（dd_soft） | 新实验用 dd_soft；旧结果用 pilot_only 复现 |
| 旧 PPO/RL、短/中记忆 profile 计划 | docs/research_route_v4_20261106.md | 历史记录；不是当前默认下一步 |

## 3. 全部文档（按类别）

### 当前规范（4）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/document_index_20261106.md](document_index_20261106.md) | 当前 | 唯一事实源/索引 |
| [docs/protocols/confirmation_criteria_v4_draft_20261106.md](protocols/confirmation_criteria_v4_draft_20261106.md) | 当前 | 唯一事实源/索引 |
| [docs/protocols/research_route_v4_20261106.json](protocols/research_route_v4_20261106.json) | 当前 | 唯一事实源/索引 |
| [docs/research_route_v4_20261106.md](research_route_v4_20261106.md) | 当前 | 唯一事实源/索引 |

### 已被取代（3）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/protocols/confirmation_criteria_draft_20261105.md](protocols/confirmation_criteria_draft_20261105.md) | 取代 | v4 取代；保留原语义 |
| [docs/protocols/research_route_v3_20261103.json](protocols/research_route_v3_20261103.json) | 取代 | v4 取代；保留原语义 |
| [docs/research_route_v3_20261103.md](research_route_v3_20261103.md) | 取代 | v4 取代；保留原语义 |

### 路线/计划/进展（109）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/abc_replay_ac_3s_30f_findings.md](abc_replay_ac_3s_30f_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/abc_replay_ac_5s_60f_findings.md](abc_replay_ac_5s_60f_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/abc_replay_smoke_findings.md](abc_replay_smoke_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/adaptability_audit_plan_20261103.md](adaptability_audit_plan_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/adaptability_long_anchor_stage_20261103.md](adaptability_long_anchor_stage_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/calibrated_compare_plan_20261101.md](calibrated_compare_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/calibrated_compare_stage_20261101.md](calibrated_compare_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_capacity_stage_20261102.md](channel_memory_capacity_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_contiguous_stage_20261102.md](channel_memory_contiguous_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_fresh_stage_20261102.md](channel_memory_fresh_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_history_plan_20261102.md](channel_memory_history_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_history_stage_20261102.md](channel_memory_history_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_plan_20261102.md](channel_memory_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_bridge_plan_20261102.md](channel_memory_rl_bridge_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_bridge_stage_20261102.md](channel_memory_rl_bridge_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_continuous_confirmation_plan_20261102.md](channel_memory_rl_continuous_confirmation_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_continuous_confirmation_stage_20261102.md](channel_memory_rl_continuous_confirmation_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_continuous_plan_20261102.md](channel_memory_rl_continuous_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_continuous_stage_20261102.md](channel_memory_rl_continuous_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_cross_frame_observation_plan_20261102.md](channel_memory_rl_cross_frame_observation_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_protocol_plan_20261102.md](channel_memory_rl_protocol_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_rl_protocol_stage_20261102.md](channel_memory_rl_protocol_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_scale_training_audit_20261102.md](channel_memory_scale_training_audit_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_scale_training_plan_20261102.md](channel_memory_scale_training_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_memory_scale_training_stage_20261102.md](channel_memory_scale_training_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_capacity_plan_20261102.md](channel_state_capacity_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_future_plan_20261102.md](channel_state_future_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_future_stage_20261102.md](channel_state_future_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_geometry_plan_20261102.md](channel_state_geometry_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_geometry_stage_20261102.md](channel_state_geometry_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_long_age_plan_20261102.md](channel_state_long_age_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_long_age_stage_20261102.md](channel_state_long_age_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_long_age_training_plan_20261102.md](channel_state_long_age_training_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_long_age_training_stage_20261102.md](channel_state_long_age_training_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_map_plan_20261102.md](channel_state_map_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_map_stage_20261102.md](channel_state_map_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_retention_plan_20261102.md](channel_state_retention_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_retention_stage_20261102.md](channel_state_retention_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_tracking_plan_20261102.md](channel_state_tracking_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_tracking_stage_20261102.md](channel_state_tracking_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_transfer_plan_20261102.md](channel_state_transfer_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_transfer_stage_20261102.md](channel_state_transfer_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_window_plan_20261102.md](channel_state_window_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/channel_state_window_stage_20261102.md](channel_state_window_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/cross_frame_offline_pair_stage_20261103.md](cross_frame_offline_pair_stage_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/cross_frame_offline_training_plan_20261103.md](cross_frame_offline_training_plan_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/cross_frame_pilot_history_stage_20261103.md](cross_frame_pilot_history_stage_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/cross_frame_stage_20261103.md](cross_frame_stage_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/detection_subspace_implementation_20261103.md](detection_subspace_implementation_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/deterministic_audit_implementation_20261103.md](deterministic_audit_implementation_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/dfe_known_feedback_plan_20261101.md](dfe_known_feedback_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/dfe_known_feedback_stage_20261101.md](dfe_known_feedback_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/dfe_rls_calibration_plan_20261101.md](dfe_rls_calibration_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/dfe_rls_calibration_stage_20261101.md](dfe_rls_calibration_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/eme_joint_state_tracking_findings.md](eme_joint_state_tracking_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/eme_long_memory_v2_phase_prior_findings.md](eme_long_memory_v2_phase_prior_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/eme_long_memory_v2_warmstart_findings.md](eme_long_memory_v2_warmstart_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/eme_offline_training_findings.md](eme_offline_training_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/eme_online_meta_adaptation_findings.md](eme_online_meta_adaptation_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/eme_online_rl_findings.md](eme_online_rl_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/eme_slow_drift_gate2_findings.md](eme_slow_drift_gate2_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/eme_two_layer_research_route.md](eme_two_layer_research_route.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/error_geometry_plan_20261101.md](error_geometry_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/error_geometry_stage_20261101.md](error_geometry_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/existing_phase_capacity_plan_20261101.md](existing_phase_capacity_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/existing_phase_capacity_stage_20261101.md](existing_phase_capacity_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/full_rx_transfer_plan_20261101.md](full_rx_transfer_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/full_rx_transfer_stage_20261101.md](full_rx_transfer_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/head_transfer_plan_20261101.md](head_transfer_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/head_transfer_stage_20261101.md](head_transfer_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/joint_pilot_state_stage_20261103.md](joint_pilot_state_stage_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/linear_baseline_calibration_plan_20261101.md](linear_baseline_calibration_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/linear_baseline_calibration_stage_20261101.md](linear_baseline_calibration_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/low_dimensional_mechanism_plan_20261103.md](low_dimensional_mechanism_plan_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/offline_equalizer_training_and_baseline_20261105.md](offline_equalizer_training_and_baseline_20261105.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/online_adaptation_progress_20261106.md](online_adaptation_progress_20261106.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/online_gain_methods_and_horizon_validation_20261105.md](online_gain_methods_and_horizon_validation_20261105.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/online_multiscale_peft_stage_20261011.md](online_multiscale_peft_stage_20261011.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/online_peft_phase_trend_head_route.md](online_peft_phase_trend_head_route.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/peft_within_frame_audit_20261028.md](peft_within_frame_audit_20261028.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/persistent_prefix_plan_20261102.md](persistent_prefix_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/persistent_prefix_stage_20261102.md](persistent_prefix_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/phase_transfer_plan_20261101.md](phase_transfer_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/phase_transfer_stage_20261101.md](phase_transfer_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/pilot_online_adaptation_stage_plan.md](pilot_online_adaptation_stage_plan.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/pilot_peft_stage_20261029.md](pilot_peft_stage_20261029.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/pilot_replay_protocol_audit_20261014.md](pilot_replay_protocol_audit_20261014.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/pilot_state_transfer_plan_20261102.md](pilot_state_transfer_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/pilot_state_transfer_stage_20261102.md](pilot_state_transfer_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/pilot_transfer_plan_20261101.md](pilot_transfer_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/pilot_transfer_stage_20261101.md](pilot_transfer_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/prefix_task_plan_20261102.md](prefix_task_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/prefix_task_stage_20261102.md](prefix_task_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/prefix_transfer_plan_20261102.md](prefix_transfer_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/prefix_transfer_stage_20261102.md](prefix_transfer_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/prefix_window_alignment_plan_20261102.md](prefix_window_alignment_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/prefix_window_alignment_stage_20261102.md](prefix_window_alignment_stage_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/receiver_input_candidate_plan_20261101.md](receiver_input_candidate_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/receiver_input_candidate_stage_20261101.md](receiver_input_candidate_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/receiver_visibility_plan_20261101.md](receiver_visibility_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/receiver_visibility_stage_20261101.md](receiver_visibility_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/rl_peft_execution_audit_20261029.md](rl_peft_execution_audit_20261029.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/rls_peft_20261002_findings.md](rls_peft_20261002_findings.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/shared_neural_pilot_stage_20261103.md](shared_neural_pilot_stage_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/shared_pilot_fixed_actions_stage_20261103.md](shared_pilot_fixed_actions_stage_20261103.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/symbol_step_response_plan_20261102.md](symbol_step_response_plan_20261102.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/symbol_transfer_plan_20261101.md](symbol_transfer_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/symbol_transfer_stage_20261101.md](symbol_transfer_stage_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |
| [docs/transfer_gradient_diagnostic_plan_20261101.md](transfer_gradient_diagnostic_plan_20261101.md) | 历史 | 开发记录；当前规范以 v4 为准 |

### 协议/判据（1）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/protocols/confirmation_physical_v1_20261106.md](protocols/confirmation_physical_v1_20261106.md) | 历史 | 旧协议；确认前以 v4 草案为准 |

### 结果归档（73）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/results/adapt_replay_dev_16f_v1_20261106/README.md](results/adapt_replay_dev_16f_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/bayes_head_diag_v1_20261106/README.md](results/bayes_head_diag_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/bayesian_cir_diag_v1_20261106/README.md](results/bayesian_cir_diag_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/cir_learned_map_v1_20261106/README.md](results/cir_learned_map_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/cumulative_radius_dev_16f_v1_20261106/README.md](results/cumulative_radius_dev_16f_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/data_aided_cir_diag_v1_20261106/README.md](results/data_aided_cir_diag_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/data_aided_elbo_diag_v1_20261106/README.md](results/data_aided_elbo_diag_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_adapt_loss_alignment_v1_20261105/README.md](results/detection_subspace_adapt_loss_alignment_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_frozen_error_outer_v1_20261105/README.md](results/detection_subspace_frozen_error_outer_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v10_context_update_v1_20261104/README.md](results/detection_subspace_long_v10_context_update_v1_20261104/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v11_smooth_ber_outer_v1_20261104/README.md](results/detection_subspace_long_v11_smooth_ber_outer_v1_20261104/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v16_dim12_frozen_error_v1_20261105/README.md](results/detection_subspace_long_v16_dim12_frozen_error_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v17_dim12_gradient_transport_v1_20261105/README.md](results/detection_subspace_long_v17_dim12_gradient_transport_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v18_dilated127_audit_v1_20261105/README.md](results/detection_subspace_long_v18_dilated127_audit_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v19_dilated127_residual_v1_20261105/README.md](results/detection_subspace_long_v19_dilated127_residual_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v1_20261103/README.md](results/detection_subspace_long_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v20_pilot_context_adapter_v1_20261105/README.md](results/detection_subspace_long_v20_pilot_context_adapter_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v20_pilot_context_adapter_v1_20261105/actual_forward_v1/README.md](results/detection_subspace_long_v20_pilot_context_adapter_v1_20261105/actual_forward_v1/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v21_pilot_context_history1_v1_20261105/README.md](results/detection_subspace_long_v21_pilot_context_history1_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v21_pilot_context_history1_v1_20261105/actual_forward_v1/README.md](results/detection_subspace_long_v21_pilot_context_history1_v1_20261105/actual_forward_v1/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v22_pilot_context_history1_crossseed_v1_20261105/README.md](results/detection_subspace_long_v22_pilot_context_history1_crossseed_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v22_pilot_context_history1_crossseed_v1_20261105/actual_forward_v1/README.md](results/detection_subspace_long_v22_pilot_context_history1_crossseed_v1_20261105/actual_forward_v1/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v23_gauss_newton_numerical_v1_20261105/README.md](results/detection_subspace_long_v23_gauss_newton_numerical_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v24_gauss_newton_meta_v1_20261105/README.md](results/detection_subspace_long_v24_gauss_newton_meta_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v25_gauss_newton_online_gain_v1_20261105/README.md](results/detection_subspace_long_v25_gauss_newton_online_gain_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v26_gauss_newton_proximal_audit_v1_20261105/README.md](results/detection_subspace_long_v26_gauss_newton_proximal_audit_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v27_gauss_newton_online_gain_proximal001_v1_20261105/README.md](results/detection_subspace_long_v27_gauss_newton_online_gain_proximal001_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v2_reset_v1_20261103/README.md](results/detection_subspace_long_v2_reset_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v3_decay_v1_20261103/README.md](results/detection_subspace_long_v3_decay_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v4_tempered_adapt_v1_20261103/README.md](results/detection_subspace_long_v4_tempered_adapt_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v5_temporal_basis_v1_20261103/README.md](results/detection_subspace_long_v5_temporal_basis_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v6_adapt_history8_v1_20261103/README.md](results/detection_subspace_long_v6_adapt_history8_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v6_gate_alignment_v1_20261103/README.md](results/detection_subspace_long_v6_gate_alignment_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v6_gradient_alignment_v1_20261103/README.md](results/detection_subspace_long_v6_gradient_alignment_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v6_temperature_sweep_v1_20261103/README.md](results/detection_subspace_long_v6_temperature_sweep_v1_20261103/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v7_adapt_temperature2_v1_20261104/README.md](results/detection_subspace_long_v7_adapt_temperature2_v1_20261104/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v8_surrogate_gate_v1_20261104/README.md](results/detection_subspace_long_v8_surrogate_gate_v1_20261104/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_long_v9_gradient_transport_v1_20261104/README.md](results/detection_subspace_long_v9_gradient_transport_v1_20261104/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_paired_online_gain_v1_20261105/README.md](results/detection_subspace_paired_online_gain_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_v27_bayesian_pilot_rls_screen_v1_20261105/README.md](results/detection_subspace_v27_bayesian_pilot_rls_screen_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/detection_subspace_v7_gate_diagnostics_20261104/README.md](results/detection_subspace_v7_gate_diagnostics_20261104/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/full_head_adapt_replay_h4_dev_v1_20261106/README.md](results/full_head_adapt_replay_h4_dev_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/full_head_radius2_dev_v1_20261106/README.md](results/full_head_radius2_dev_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/full_head_reward_gate_h4_smoke_v1_20261106/README.md](results/full_head_reward_gate_h4_smoke_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/full_head_softber_supervised_train32_v1_20261106/README.md](results/full_head_softber_supervised_train32_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/full_head_softber_train32_v1_20261106/README.md](results/full_head_softber_train32_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/full_head_trajectory_dev8_v1_20261106/README.md](results/full_head_trajectory_dev8_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/full_head_trajectory_dev8_v1_20261106/head_gradient_alignment_diagnostic.md](results/full_head_trajectory_dev8_v1_20261106/head_gradient_alignment_diagnostic.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/full_head_trajectory_train32_v1_20261106/README.md](results/full_head_trajectory_train32_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/head_adapt_fisher_dev_v1_20261106/README.md](results/head_adapt_fisher_dev_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/head_fisher_rank1_gn_smoke_v1_20261106/README.md](results/head_fisher_rank1_gn_smoke_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/head_fisher_rank1_smoke_v1_20261106/README.md](results/head_fisher_rank1_smoke_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/method_survey_20261106/README.md](results/method_survey_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/offline_saturation_20261106/README.md](results/offline_saturation_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/parameter_group_direction_dev_v1_20261106/README.md](results/parameter_group_direction_dev_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/persistent_adam_dev_16f_v1_20261106/README.md](results/persistent_adam_dev_16f_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/persistent_adam_dev_32f_v1_20261106/README.md](results/persistent_adam_dev_32f_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/pilot_validation_risk_gate_replacement_v1_20261105/README.md](results/pilot_validation_risk_gate_replacement_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/pilot_validation_risk_gate_v1_20261104/README.md](results/pilot_validation_risk_gate_v1_20261104/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/power_error_composition_20261106/README.md](results/power_error_composition_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/rank1_gn_history_eval_v1_20261106/README.md](results/rank1_gn_history_eval_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/rank1_gn_meta_fisher_eval32_rep_v1_20261106/README.md](results/rank1_gn_meta_fisher_eval32_rep_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/rank1_gn_meta_fisher_eval32_v1_20261106/README.md](results/rank1_gn_meta_fisher_eval32_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/rank1_gn_meta_smoke_v1_20261106/README.md](results/rank1_gn_meta_smoke_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/rank1_state_filter_dev_v1_20261106/README.md](results/rank1_state_filter_dev_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/strong_checkpoint_pilot_transfer_meta32_20261105/README.md](results/strong_checkpoint_pilot_transfer_meta32_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/strong_offline_checkpoint_rls_smoke_20261105/README.md](results/strong_offline_checkpoint_rls_smoke_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/strong_offline_rls_onewindow_32f_dev_20261105/README.md](results/strong_offline_rls_onewindow_32f_dev_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/strong_offline_rls_onewindow_64f_dev_20261105/README.md](results/strong_offline_rls_onewindow_64f_dev_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/subspace_v27_checkpoint_replay_v1_20261105/README.md](results/subspace_v27_checkpoint_replay_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/tail_alignment_v1_20261106/README.md](results/tail_alignment_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/trajectory_transfer_dev_16f_v1_20261105/README.md](results/trajectory_transfer_dev_16f_v1_20261105/README.md) | 结果 | 原语义；不按 v4 追认 |
| [docs/results/trajectory_transfer_or_gate_dev_16f_v1_20261106/README.md](results/trajectory_transfer_or_gate_dev_16f_v1_20261106/README.md) | 结果 | 原语义；不按 v4 追认 |

### 流程文档（6）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/superpowers/plans/2026-08-25-eme-measurement-constrained-channel.md](superpowers/plans/2026-08-25-eme-measurement-constrained-channel.md) | 流程 | 过程/设计记录 |
| [docs/superpowers/plans/2026-08-26-eme-frozen-profile-passthrough.md](superpowers/plans/2026-08-26-eme-frozen-profile-passthrough.md) | 流程 | 过程/设计记录 |
| [docs/superpowers/plans/2026-08-31-eme-heldout-online-recovery.md](superpowers/plans/2026-08-31-eme-heldout-online-recovery.md) | 流程 | 过程/设计记录 |
| [docs/superpowers/plans/2026-08-31-eme-online-meta-adaptation.md](superpowers/plans/2026-08-31-eme-online-meta-adaptation.md) | 流程 | 过程/设计记录 |
| [docs/superpowers/specs/2026-08-25-eme-measurement-constrained-channel-design.md](superpowers/specs/2026-08-25-eme-measurement-constrained-channel-design.md) | 流程 | 过程/设计记录 |
| [docs/superpowers/specs/2026-08-31-eme-online-meta-adaptation-design.md](superpowers/specs/2026-08-31-eme-online-meta-adaptation-design.md) | 流程 | 过程/设计记录 |

### 其他文档（44）

| 文档 | 状态 | 说明 |
|---|---|---|
| [docs/dual_view_pilot_consistency_replay_20261010.md](dual_view_pilot_consistency_replay_20261010.md) | 历史 | 保留原语义 |
| [docs/eme_action_delay_diagnostic.md](eme_action_delay_diagnostic.md) | 历史 | 保留原语义 |
| [docs/eme_ber_root_cause_and_fix.md](eme_ber_root_cause_and_fix.md) | 历史 | 保留原语义 |
| [docs/eme_channel_parameters_and_literature.md](eme_channel_parameters_and_literature.md) | 历史 | 保留原语义 |
| [docs/eme_condition_boundary_diagnostic.md](eme_condition_boundary_diagnostic.md) | 历史 | 保留原语义 |
| [docs/eme_guarded_online_state_recovery_results.md](eme_guarded_online_state_recovery_results.md) | 历史 | 保留原语义 |
| [docs/eme_long_episode_diagnostic_200f.md](eme_long_episode_diagnostic_200f.md) | 历史 | 保留原语义 |
| [docs/eme_long_memory_v2研究设计.md](eme_long_memory_v2研究设计.md) | 历史 | 保留原语义 |
| [docs/eme_measurement_channel_calibration.md](eme_measurement_channel_calibration.md) | 历史 | 保留原语义 |
| [docs/eme_online_microtuning_boundary.md](eme_online_microtuning_boundary.md) | 历史 | 保留原语义 |
| [docs/eme_online_recovery_results.md](eme_online_recovery_results.md) | 历史 | 保留原语义 |
| [docs/eme_online_stability_calibration.md](eme_online_stability_calibration.md) | 历史 | 保留原语义 |
| [docs/eme_pilot_replay_diagnostic.md](eme_pilot_replay_diagnostic.md) | 历史 | 保留原语义 |
| [docs/eme_pilot_resource_ablation.md](eme_pilot_resource_ablation.md) | 历史 | 保留原语义 |
| [docs/eme_reward_pilot_split_and_update_rate.md](eme_reward_pilot_split_and_update_rate.md) | 历史 | 保留原语义 |
| [docs/eme_second_research_point_design.md](eme_second_research_point_design.md) | 历史 | 保留原语义 |
| [docs/eme_slow_drift_physical_rl_plan.md](eme_slow_drift_physical_rl_plan.md) | 历史 | 保留原语义 |
| [docs/first_principles_20261101.md](first_principles_20261101.md) | 历史 | 保留原语义 |
| [docs/formal_signal_guard_logitcap01_gap0_10db_seed77_81_60f.md](formal_signal_guard_logitcap01_gap0_10db_seed77_81_60f.md) | 历史 | 保留原语义 |
| [docs/formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f.md](formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f.md) | 历史 | 保留原语义 |
| [docs/formal_signal_guard_logitcap01_signalmin0005_gap0_10db_seed77_81_60f.md](formal_signal_guard_logitcap01_signalmin0005_gap0_10db_seed77_81_60f.md) | 历史 | 保留原语义 |
| [docs/four_way_replay_status.md](four_way_replay_status.md) | 历史 | 保留原语义 |
| [docs/linear_operator_fft_20261101.md](linear_operator_fft_20261101.md) | 历史 | 保留原语义 |
| [docs/online_logit_guard_replay_20261015.md](online_logit_guard_replay_20261015.md) | 历史 | 保留原语义 |
| [docs/pftnet与第二研究点衔接分析.md](pftnet与第二研究点衔接分析.md) | 历史 | 保留原语义 |
| [docs/phase_modulation_geometry_20261101.md](phase_modulation_geometry_20261101.md) | 历史 | 保留原语义 |
| [docs/pilot_holdout_waveform_isolation_20261017.md](pilot_holdout_waveform_isolation_20261017.md) | 历史 | 保留原语义 |
| [docs/pilot_only_margin_consensus_20261016.md](pilot_only_margin_consensus_20261016.md) | 历史 | 保留原语义 |
| [docs/pilot_peft_group_screen_20261013.md](pilot_peft_group_screen_20261013.md) | 历史 | 保留原语义 |
| [docs/pilot_position_scan_and_signal_reconstruction_20261010.md](pilot_position_scan_and_signal_reconstruction_20261010.md) | 历史 | 保留原语义 |
| [docs/pilot_reliability_oof_20261013.md](pilot_reliability_oof_20261013.md) | 历史 | 保留原语义 |
| [docs/rl_peft_identity_control_20261030.md](rl_peft_identity_control_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_main_comparison_l1e3_20261030.md](rl_peft_main_comparison_l1e3_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_moderate_replay_20261030.md](rl_peft_moderate_replay_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_moderate_selection_followup_20261030.md](rl_peft_moderate_selection_followup_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_multisnr_lr1e3_20261030.md](rl_peft_multisnr_lr1e3_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_pilot_multiscale_ranking_strong_gap0_20261030.md](rl_peft_pilot_multiscale_ranking_strong_gap0_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_pilot_state_embedding_update16_20261030.md](rl_peft_pilot_state_embedding_update16_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_policy_probability_lr_20261030.md](rl_peft_policy_probability_lr_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_reward_pilot_length_replay_20261030.md](rl_peft_reward_pilot_length_replay_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_strong_gap_identity_control_20261030.md](rl_peft_strong_gap_identity_control_20261030.md) | 历史 | 保留原语义 |
| [docs/rl_peft_update_radius_replay_20261030.md](rl_peft_update_radius_replay_20261030.md) | 历史 | 保留原语义 |
| [docs/state_conditioned_peft_replay_20261011.md](state_conditioned_peft_replay_20261011.md) | 历史 | 保留原语义 |
| [docs/state_mapping_scope_replay_20261014.md](state_mapping_scope_replay_20261014.md) | 历史 | 保留原语义 |

## 4. 维护规则

1. 新增结果归档使用 `docs/results/<name>_<date>/README.md`，写目的、协议、命令、数值、边界与指纹。
2. 新的路线/判据变更先改 v4 单一事实源，再在结果归档中引用；不静默改写旧结果。
3. 旧文档只加取代标注，不删除、不改写数值；索引由脚本重新生成。
4. 正式确认前冻结判据与 checkpoint；开发集不得反复当作确认集。
