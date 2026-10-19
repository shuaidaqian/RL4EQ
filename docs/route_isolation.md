# 研究路线隔离

仓库维护两条独立研究路线，使用不同 Git 分支，不交叉 cherry-pick 主入口或规范文档。

## 物理状态跟踪主线

- 分支：`codex/physical-state-tracking`
- 目标：离线预训练长记忆 ISI 基础均衡器；在线冻结神经主干，仅用 prefix Adapt Pilot 跟踪 CIR、CFO、慢相位及跨帧 soft-tail belief。
- 默认配置：`configs/level_b_state_tracking.json`
- 权威设计与验收协议：`docs/state_tracking_receiver_design.md`、`docs/state_tracking_validation_protocol.md`
- Data 标签只用于事后硬 BER。未经独立轨迹验证的 Pilot 风险分数不能驱动状态提交或动作选择。

## 在线 PEFT 历史路线

- 分支：`codex/online-peft`
- 该分支保留 Pilot 驱动 PEFT 与 PPO 实现，作为历史复现和对照。
- PEFT/PPO 路线不作为当前默认入口；其代码与实验记录不合入物理状态跟踪分支。

## 分支操作约定

开展某一路线前先确认 `git branch --show-current`。物理状态研究只在 `codex/physical-state-tracking` 进行；PEFT/PPO 历史复现只在 `codex/online-peft` 进行。共同修复需分别评估并按需独立提交，避免用一条路线的入口覆盖另一条。
