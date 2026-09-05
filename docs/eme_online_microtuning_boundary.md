# EME 在线微调价值与因果边界

## 1. 当前研究问题

第二研究点不是简单比较一个离线神经网络和一个在线神经网络的最终 BER，而是要回答：

> 当 Level B 极端长记忆信道的慢相位、residual CFO 或跨帧状态发生变化时，离线模型固定后，前缀 Adapt Pilot 驱动的受限参数更新能否恢复离线模型失去的性能？

因此必须把三类增益分开：

1. **当前 Pilot 条件化**：网络读取当前帧 Adapt Pilot，重新计算 phase/CFO 条件，但参数不变；
2. **在线物理状态更新**：根据 Adapt Pilot 更新 CIR 或慢相位状态；
3. **在线参数微调**：根据 Adapt Pilot 的自监督损失更新 phase、FiLM、Adapter、LoRA 或 head 等受限 PEFT 参数，Reward Pilot 只负责验收和回滚。

`Frozen Offline NN` 是在线增量的唯一离线接收机对照。它与 `Pilot-Driven Online Adaptation` 使用相同的 checkpoint、信道轨迹、Pilot 切分和 soft-tail 递推；Frozen 固定 acquisition 条件，Online 才读取当前帧 Pilot 并更新物理状态和 PEFT 参数。

## 2. 新增的严格参数微调消融

`compare.py` 增加了：

```powershell
--online-condition-source acquisition
```

该模式让 online 方法也固定使用 acquisition 条件，不读取当前帧 Pilot 的 phase 条件；它仍然可以从 Adapt Pilot 计算梯度，并由 Reward Pilot 决定是否保留参数更新。它的作用是测量“参数在线微调本身”的增量，而不是替代主结果。

允许的条件源为：

- `acquisition`：严格离线条件，适合参数微调因果消融；
- `pilot_phase`：使用当前 Adapt Pilot 的 phase 特征，但 CIR 固定；
- `pilot_cir_phase`：使用当前 Pilot 的 phase 特征，并允许与 Pilot CIR 跟踪链配合。

在线路径不因该选项读取 Data 标签。Data 标签仅用于最终仿真 BER 统计；Adapt Pilot 用于更新，Reward Pilot 用于更新后的 reward、验收和回滚。

## 3. 通俗解释

离线训练先用完整帧和 Data 标签，把网络训练成“知道这类长回波信道通常怎么均衡”的基础接收机。实际运行时，每一帧前缀里的 Adapt Pilot 是接收端已知答案的一小段信号，因此可以用它计算当前网络的错误，并只调整 `phase` 这类很少的 PEFT 参数。

一次在线更新的过程是：先保存旧参数；用 Adapt Pilot 做一小步梯度更新；再把独立的 Reward Pilot 分成两个时间子窗口，检查更新后是否在两个子窗口都变好。只要有一个子窗口变差，或者总体没有达到最小改善，就恢复旧参数；下一帧发现上一次更新造成恶化时也会恢复。Data 区域没有标签，不参与在线训练、选动作或回滚，只在实验结束后统计 BER。

所以 Frozen Offline NN 是“离线学会基础能力、运行时状态也冻结”，Online 是“从同一个离线起点出发，根据每帧 Pilot 恢复当前状态并小幅修正参数”。论文需要证明的是同一信道、同一 checkpoint、同一 Pilot 和同一帧轨迹下，在线恢复与微调能为 Frozen 带来稳定的额外收益。

## 3. 已发现的比较错误及修正

早期比较中，Frozen 神经方法使用 `soft-tail` 直接替换，online 方法使用配置的平滑系数 `tail_update_alpha`。在没有任何 PEFT 更新被接受时，两者仍可能产生不同 BER，这会把跨帧状态递推差异误报为在线微调收益。

现在所有神经方法都通过 `_update_receiver_tail()` 使用同一条规则。主配置的 `tail_update_alpha=0.5` 由配置统一传递；该修正后的矩阵必须重新运行，旧的 Frozen/Online 数值不能与新矩阵混合。

## 4. 阶段性探针结论

在当前 checkpoint、Level B `heldout_edge`、`max_delay=116`、prefix Pilot=128、Reward Pilot=64 的小样本探针中：

- 当前 Pilot 条件化的 Frozen NN 在 10/15 dB 已达到约 `0.688%/0.112%`；
- 默认 online（同时使用当前 Pilot phase 条件）与 Frozen 接近，说明单纯增加 Reward Pilot 或放宽 PEFT 门限并不能自动产生显著额外收益；
- 固定 acquisition 条件、只更新 phase 参数时，较强的每帧更新探针把约 `53.7%/55.3%` 降到约 `46.0%/45.6%`，证明 Adapt Pilot 驱动的参数更新确实能修正一部分状态失配，但 phase-only 不是恢复稀疏长回波结构的充分参数对象；
- 更新间隔为 4 帧时会出现“某帧接近正确、下一帧严重失配”的周期性现象，说明在线更新率必须与慢相位状态变化速度匹配，不能固定沿用一个未经验证的间隔；
- 该参数微调探针仍远未达到主性能目标，不能作为最终主结果。它的价值在于建立了可审计的因果边界，并指出最终主线应采用更合适的低维 PEFT 对象和逐帧安全调度。

## 5. 最终实验顺序

1. 在相同 seed/信道轨迹上比较 `Frozen Offline NN` 与完整 Online，并报告 accepted、rollback、Reward Pilot loss 和 Data BER 的配对变化；
2. 在相同 seed/信道轨迹上比较逐帧和窗口更新策略，并报告 Online 相对 Frozen 的增量；
3. 固定一个表现稳定的更新对象后，再比较 `phase`、`conditioner_film`、`adapter_lora` 和 `head`，不把多个对象同时改变；
4. 只有当 Reward Pilot 的窗口统计能稳定排序在线动作，才考虑 Contextual Bandit；在此之前不把 Bandit 作为主贡献；
5. 论文中将“在线微调”表述为 Pilot-only、自监督、受限参数更新和安全回滚机制，并同时报告其适用边界，不能把当前 Pilot 条件化收益全部归因于 PEFT。
