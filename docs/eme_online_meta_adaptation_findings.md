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

## 三层路线复核（2026-09-05）

当前实现已经把三层研究对象分开：

1. 离线训练默认使用整帧等权 `BCE_all`，模型输入仍通过 `receiver_view()` 隐藏
   Reward Pilot 和 Data 的发送符号；
2. 元学习保留为必要性实验，并新增 `build_meta_necessity_report`，要求与普通
   Pilot-SGD 在同一 checkpoint、轨迹、预算下配对比较后期帧和 `heldout_edge`；
3. 在线调度使用安全 Contextual Bandit，动作只选择当前模型存在的 PEFT 组，
   Reward Pilot loss 改善扣除更新成本和回滚惩罚，Data 标签不进入 Bandit。

新增长处延迟诊断脚本 `scripts/diagnose_action_delay.py` 汇总 horizon
`0/1/2/4/8`。在 `logs/eme_online_bandit_smoke_20260905` 的 116 符号记忆、
10 dB、8 帧 EME smoke 中，`delayed_effect_detected=false`，当前证据支持继续
使用 Contextual Bandit；8 帧单 seed 不是升级 Recurrent Double DQN 或声称性能
增益的充分证据。

## 32 步初始化加元训练的正式 Level B 矩阵

训练流程为：先使用 `stage all --steps 32` 得到离线初始化，再使用 `stage online_meta --steps 32`、每个 episode 连续 4 帧进行序列元训练。使用的 checkpoint 和完整矩阵为：

```text
pretrained/eme_meta_from_offline_32/model_best.pt
logs/eme_meta_from_offline_32_main5x60/
```

主矩阵使用固定 `eme_long_memory_v2`、`max_delay=116`、prefix Pilot、5 个 seed、每个 seed 60 帧、SNR `0/5/10/15 dB`。每个方法每个 SNR 均有 300 条逐帧记录，全部 19200 条记录的 `(method, SNR, seed, frame)` 唯一键检查通过。

| SNR | CFO+DD LMMSE-FIR | CFO+DD DFE-RLS | Frozen Offline NN | Pilot Online |
|---:|---:|---:|---:|---:|
| 0 dB | 0.452857 | 0.447318 | 0.208590 | 0.211421 |
| 5 dB | 0.316760 | 0.303754 | 0.089446 | 0.091298 |
| 10 dB | 0.185510 | 0.181674 | 0.047281 | 0.047351 |
| 15 dB | 0.124959 | 0.121466 | 0.035562 | 0.035543 |

相对于两个传统方法中较强者，Pilot Online 的 BER 降幅约为 `52.7%/69.9%/73.9%/70.7%`（对应 `0/5/10/15 dB`）。因此当前正式证据支持：在线 Pilot 方法在 Level B 长记忆、残余 CFO 和慢相位条件下，四个主 SNR 均明显优于公平传统 baseline，并且 60 帧递推保持稳定。

但是，Pilot Online 相对 Frozen Offline NN 的差异为：0 dB 略差、5 dB 略差、10 dB 基本持平、15 dB 基本持平。Online 的前 5 帧/后 5 帧 Data BER 为 `0.200402/0.203259`、`0.0752679/0.0887054`、`0.0390179/0.0526786`、`0.0289732/0.0395536`。因此不能声称在线增益随帧数增加而持续变大；下一阶段若要证明在线研究点的额外价值，必须增加同一冻结 profile 内、离线未见但物理合法的慢状态失配测试，并让训练外层目标显式优化跨帧恢复，而不是继续扩大匹配分布训练。
