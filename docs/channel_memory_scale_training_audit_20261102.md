# 64步尺度训练完成，独立机制验证前审计通过

按[固定计划](channel_memory_scale_training_plan_20261102.md)，A/B各64步、offline_train 98000–98063已完成。B仅训练34个尺度参数，不改任何旧权重；A保持固定尺度。**这里只确认训练与审计完成，不宣称在线收益或总目标完成。**

## 审计事实

- A/B的64个唯一训练样本，seed/SNR/物理年龄/当前及未来帧号/缓存键/实际波形SHA/信道状态逐项匹配。
- 所有缓存完整协议与配置一致，离线监督标记、帧号和内容指纹一致。
- 两分支的234个记忆参数均严格为零，旧权重逐项等于共同源。A的尺度全零，B的尺度确实变化。
- 零记忆残差恒零，因此B的离线尺度变化不会改变未更新Frozen；此前零输出与梯度测试已验证这一结构。
- 离线内循环仍只读Adapt；Data仅用于更新后外层监督。没有从smoke权重接续，也没有按留出选择checkpoint。

源SHA `919ca4c6e99c5c44519c1b381e44baa4b923e204c1daccc56da339b50a6ab71e`。

控制A：`pretrained/channel_memory_map_control64_20261102/model_best.pt`，SHA `153ccc68515fe77996f822924f0056d25a283ca0fd7b185c19a030bfe2871f08`。

学习B：`pretrained/channel_memory_map_post64_20261102/model_best.pt`，SHA `dcc7c680cb84203b65029e6c15f2d45cda535aef293f106475e6667b3c11bf58`。

A研究墙钟337.58秒，缓存40命中/24缺失；B52.25秒，64命中/0缺失。两个CPU准备段另耗239.41/238.77秒，与A并行，不能相加作总墙钟。这些不是每帧生产时延。

## 接续验证

预注册heldout_edge 99001–99005、四SNR、每条burn-in1023帧后评分8帧，接受后跨帧保留，输出到 `logs/channel_memory_map_four_arm64_5s8f_20261102`。这是约105秒处开始更新的四臂机制，不能替代从首帧连续跟踪、正式窗口RL或传统主比较。

审计入口 [audit_channel_memory_training.py](../scripts/audit_channel_memory_training.py)，[机器记录](results/channel_memory_scale_training_audit_20261102.json)。其参数审计直接在实际64步终点运行通过；核心实现的全量793项与后续定向103项验证见前阶段记录。
