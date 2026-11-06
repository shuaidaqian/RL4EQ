# 匹配普通监督续训对照：soft-BER阶段

本目录保存 soft-BER trajectory 实验的匹配普通监督续训 checkpoint、训练指标和新 session 逐帧评估。它与 `full_head_softber_train32_v1_20261106` 使用相同来源离线模型、训练配置、32步预算及评估 session 76170/76171；在线评估分支为同 checkpoint 的 Frozen 与 `full_head_adam`，每个 session 连续16帧，覆盖0/5/10/15 dB。

此对照用于判断在线点估计是否只是普通监督续训带来的变化。两臂的差异不构成在线更新因果证据；因果量仍是各 checkpoint 内同波形 Online−Frozen，再比较两臂差值。每个SNR仅两个开发 session，不能用于确认或可靠统计推断。

普通监督评估 JSON 为 `eval16.json`，训练 checkpoint 与指标位于 `checkpoint/`。并列的 soft-BER trajectory 汇总和该轮结论见[trajectory结果](../full_head_softber_train32_v1_20261106/README.md)。Data标签仅用于离线监督和动作后事后评分，不用于在线更新或 Reward 接受决策。
