# Rank-1 GN 轨迹元训练梯度 smoke

## 设置

固定 Level B `eme_long_memory_v2` 长回波配置与现有 Pilot 资源，不改信道、同步或接收窗口。强离线来源 checkpoint SHA256：`8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。运行配置为 [`eme_long_memory_v2_rank1_gn_meta_smoke_v1.json`](../../../configs/eme_long_memory_v2_rank1_gn_meta_smoke_v1.json)，以两个连续帧、一个外层优化 step、0 dB 做实际 GPU 梯度 smoke。基础模型参数全部冻结，外层优化器只更新归一化的 65 维 head rank-1 basis；内循环按每帧 Adapt Pilot 做一维 GN，Reward Pilot 独立硬门控并通过既有直通代理传递元梯度，Data 只进入离线外层损失。

训练 checkpoint 为 [`model_best.pt`](checkpoint/model_best.pt)，保存的 learned basis 范数为1。初始 seeded Gaussian 方向与一步训练后方向的余弦约 `0.99997`，L2 位移约 `0.00757`，表明实际模型的两帧外层梯度能到达 basis 并产生有限更新。

来源指纹：smoke checkpoint SHA256 `d0a078f8186857d9c4362c2ee40b22f54c688975aef966117a3154dbdbee81c8`；训练配置 SHA256 `b23f532544e8fe8c498a63acd957d7ca30adb7c4bc596332cd2422b7eadc0472`；训练实现 [`pilot_transfer.py`](../../../training/pilot_transfer.py) SHA256 `537cbc0f808a27fe3b5761bb72f4de8e784ce6552cf984626511a35b4c885462`。

## 观察与限制

两个帧候选都被 Reward 硬规则拒绝。硬前向下 Online Data BCE 与 Frozen 相同（`0.46200174`）；Reward BCE 改善/退化幅度不足以触发提交。因而本次只验证“实际网络、math SDPA 后端、两帧 rank-1 GN 外层反传能运行”，没有验证任何在线 Data 增益、可接受更新率或随帧增长趋势。直通代理仅用于离线外层求导，不能当成部署时已提交的在线状态。

## 下一步

评估端仍须加载 checkpoint 中的 learned basis，并使用与训练相同的标量 GN、步幅、累计半径和 Reward 规则。随后在全新 session seed 上按0/5/10/15 dB做多帧开发评估，先检查 Reward 接受率和实际参数提交，再计算 Online−Frozen 的会话级窗口趋势。若硬门持续拒绝，下一项应校准训练期门控代理/候选尺度，但不能放松线上硬门槛，也不能使用 Data 标签做控制。
