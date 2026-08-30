# EME 在线元适配阶段性结果

## 本阶段完成内容

本阶段新增 `training/online_meta_adaptation.py`，把普通 Pilot 微调扩展为可用于真实 EME 环境的序列化一阶在线元适配训练：

```text
Adapt Pilot 内循环更新
    -> post-Adapt 前向
    -> Reward Pilot 外层目标
    -> 基础模型元更新
    -> 下一帧 soft-tail 递推
```

新增 `pretrain.py --stage online_meta` 入口。该入口使用 `CommunicationEnvironment` 生成真实 Level B 帧，使用 acquisition Pilot 估计条件，不使用真实 CIR 作为模型输入。EME 配置已加入：

- `online_meta_training=true`
- `meta_sequence_frames=4`
- `meta_inner_steps=1`
- `meta_inner_learning_rate=1e-4`
- `meta_peft_groups=["head", "conditioner_film"]`

## 信息边界

内循环只使用当前帧前缀 Adapt Pilot 的已知符号。post-Adapt 外层目标只使用留出的 Reward Pilot 标签；Data 区域不进入元训练步骤的计算图。在线 guard 版本不使用 Reward Pilot 反向更新网络，只用它判断是否接受参数更新。拒绝更新时恢复 PEFT 参数，但不回退已经形成的最新 soft tail。

## Smoke 证据

定向测试：

```text
tests/test_online_meta_adaptation.py: 6 passed
```

EME 2 步、每个 episode 4 帧的 smoke 已成功生成：

```text
pretrained/eme_online_meta_smoke/
```

该 smoke 记录了 8 条序列元训练记录，所有记录的 `condition_source` 为 `acquisition_pilot`，`data_labels_used_online` 为 `false`。各帧 inner update 的参数增量为有限小量，说明 Adapt Pilot 到 PEFT 更新链路正常。

使用该 smoke checkpoint 的 1 seed、10 dB、116 符号记忆、8 帧短对比也成功生成：

```text
logs/eme_online_meta_smoke_compare/
```

短对比中 `Offline NN only` 的 Data BER 为 `0.06069`，`Pilot-Driven Online Adaptation` 为 `0.06180`，CFO+DD 两个传统 baseline 约为 `0.106` 和 `0.104`。这组结果只能证明入口、strict-load、长记忆递推和统计链路可运行，不能作为正式论文性能，也不能据此声称在线元适配已经优于冻结离线模型。

## 发现与下一步

当前最重要的实现修正是：post-Reward 外层梯度现在可以更新基础模型，而不仅仅是 PEFT 参数；内循环仍然只更新选定 PEFT group。下一步必须进行充分的序列元训练，并在同一个冻结 Level B profile 内构造合法的未见慢状态轨迹，比较 Frozen Offline NN、普通 Pilot PEFT 和在线元适配。

如果在充分训练和有限状态失配主实验中，在线元适配仍不能稳定超过 Frozen Offline NN，则论文应报告“安全在线适配保持性能”，不能把 PPO 或更新接受率包装成在线均衡增益。
