# 迭代记录：RL-Modulated 神经块均衡器门槛式执行

时间戳：2026-08-02 18:19:07 +08:00

## 1. 本轮执行目标

用户要求不再写新 spec/plan，直接执行“门槛式自动推进，失败即停止分析”：

1. 补齐离线预训练 checkpoint。
2. 补齐真实传统 baseline。
3. 接入 RL-Modulated Neural Block Equalizer。
4. 跑 smoke 与小样本 gate。
5. 只有 gate 通过才启动正式 pilot sweep/主矩阵。

## 2. 研究路线重对齐

本轮将仓库研究合同从旧的 `Continual PPO 必须超过 Best Fixed` 调整为：

```text
RL-Modulated Neural Block Equalizer
-> 超过传统非神经、非 RL baseline
-> 每个 Level B 主配置 BER_data < 0.01
```

关键边界：

- Proposed 方法是唯一使用神经网络和 RL 的方法。
- 传统 baseline 不使用神经网络，不使用 RL。
- `Perfect-CSI Block` 与 `Fixed CG-BPSK-DD Block Detector` 只作为诊断参考，不作为主 baseline。
- Data Oracle 不恢复。
- 在线 observation、reward、动作选择、调制更新不使用 Data 标签，也不使用数据标签上界。

## 3. 代码改动

### 3.1 真实传统 baseline

新增：

```text
baseline/traditional_equalizers.py
tests/test_traditional_baselines.py
```

已实现传统非神经、非 RL 方法：

```text
LMMSE-FIR
LMS
NLMS
RLS Linear
DFE-RLS
SC-FDE-MMSE
```

测试约束：

- 输出 shape 正确。
- 每个方法报告真实 `algorithm` 名称。
- `uses_neural_network=False`。
- `uses_rl=False`。
- `shared_placeholder_kernel=False`。
- 修改 Reward/Data 隐藏标签不改变传统 baseline 输出。

### 3.2 RL 调制神经均衡器链路

新增：

```text
agent/modulation.py
agent/rl_modulator.py
training/rl_modulated_online.py
tests/test_rl_modulated_equalizer.py
```

修改：

```text
agent/unfolded_equalizer.py
agent/cir_estimator.py
online_train.py
compare.py
```

实现内容：

- `ModulationConfig` 与 `ModulationState`。
- 有界连续调制向量：
  - Adapter gates。
  - FiLM residual scale。
  - LoRA scales。
  - Head temperature。
  - Head bias。
  - Confidence threshold。
- `ContinuousModulationPolicy`。
- `ModulationObservationEncoder`，字段白名单排除 Reward/Data 标签。
- `UnfoldedEqualizer.forward(..., modulation=None)`，保持旧接口兼容。
- `run_rl_modulated_online()`。
- `run_rl_modulated_frame()`。
- `online_train.py` 切换到 RL-Modulated 在线 runner。

### 3.3 compare 接入

修改：

```text
compare.py
tests/test_evaluation_contract.py
```

新增方法组：

```text
traditional
proposed
diagnostic
main
all
```

新增 CLI：

```text
--method-group
--pilot-total
--pilot-layout
--device
```

修复/补齐：

- `--pretrained` strict-load。
- proposed 方法输出 `pretrained_loaded=True`。
- `--resume` append-safe key 包含：

```text
method, delay, snr_db, seed, frame, pilot_total, pilot_layout
```

### 3.4 离线预训练数据流

修改：

```text
training/curriculum.py
tests/test_meta_adaptation.py
```

原问题：

```python
CurriculumTrainer._step_loss()
```

此前实际训练的是加噪 identity 随机序列，且 `del phase`，没有使用 Level A/B 真实信道帧。这会产生可 strict-load 但对真实长回波信道无意义的 checkpoint。

本轮修复：

- `_step_loss()` 改为通过 `CommunicationEnvironment` 生成真实信道帧。
- 使用 `frame.rx_symbols`、`frame.bits`、`frame.model_region_ids`、`frame.tail_symbols`。
- Level A perfect phase 使用 `frame.true_cir`。
- estimated phase 使用 acquisition frame 估计 CIR。
- loss 按 Data / Adapt / Reward mask 分别计算。

## 4. 验证结果

### 4.1 全量测试

命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
94 passed in 192.18s
```

### 4.2 真实信道预训练 smoke

命令：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained\final_smoke_real_channel
```

结果：

```text
saved pretrained\final_smoke_real_channel
```

预训练 loss 记录显示已走真实信道课程阶段：

```text
cir_level_a: 1.0737 -> 0.9523
perfect_cir_level_a: 1.0391 -> 1.1369
estimated_cir_level_a: 0.9044 -> 0.9254
estimated_cir_level_b: 0.9547 -> 1.1665
```

注意：2-step smoke 只验证链路，不代表训练收敛。

### 4.3 在线 smoke

命令：

```powershell
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained\final_smoke_real_channel\model_best.pt --frames 2 --num-seeds 1 --update-interval 1 --delays 20 --snrs 10 --pilot-total 64 --pilot-layout multi_block --amp --output-dir logs\final_online_smoke_real_channel
```

结果：

```text
saved logs\final_online_smoke_real_channel
```

关键指标：

```text
mean BER_data ≈ 0.5402
frame 1 BER_data ≈ 0.5737
frame 2 BER_data ≈ 0.5067
pretrained_loaded = true
data_labels_used_online = false
```

结论：在线链路可运行，但神经接收机性能仍接近随机，不能进入 pilot sweep。

### 4.4 compare smoke

命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --pretrained pretrained\final_smoke_real_channel\model_best.pt --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 64 --pilot-layout multi_block --resume --output-dir logs\final_compare_smoke_real_channel
```

结果：

```text
saved logs\final_compare_smoke_real_channel
```

关键单帧指标：

```text
LMMSE-FIR BER_data ≈ 0.1763
LMS BER_data ≈ 0.3281
NLMS BER_data ≈ 0.5313
RLS Linear BER_data ≈ 0.1786
DFE-RLS BER_data = 0.0
SC-FDE-MMSE BER_data ≈ 0.00446

Offline NN only BER_data ≈ 0.5022
NN + Fixed Modulation BER_data ≈ 0.4911
NN + Rule Modulation BER_data ≈ 0.4777
NN + Discrete PEFT Scheduler BER_data ≈ 0.5022
RL-Modulated Neural Block Equalizer BER_data ≈ 0.5089
```

结论：

- proposed 当前没有超过传统 baseline。
- proposed 当前没有达到 `BER_data < 0.01`。
- `SC-FDE-MMSE` 与 `DFE-RLS` 在该 smoke 配置下明显强于神经 proposed。
- 因此不能启动正式 pilot sweep 或主矩阵。

### 4.5 Level B gate smoke

命令：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_level_b_gates.py --config configs/continual_ppo.json --output-dir logs\gates_level_b_smoke --frames-per-config 1 --seeds 0 --pilot-total 64 --pilot-layout multi_block
```

结果：

```text
Perfect-CIR gate_pass = true
Perfect-CIR mean_ber_data = 0.0

Best Fixed diagnostic gate_pass = true
Best Fixed diagnostic mean_ber_data ≈ 0.00260
Best Fixed diagnostic max_config_ber_data = 0.0078125

Reward/Data Spearman gate_pass = false
Spearman = NaN
num_pairs = 36
```

解释：

- 强诊断检测器可达，说明信道本身不是不可检测。
- Reward/Data Spearman 在小样本下退化为 NaN，说明当前 reward 信号对 Data 改善的排序证据不足。
- 按门槛式自动推进规则，Spearman gate 失败后必须停止，不继续 pilot sweep。

## 5. 本轮结论

本轮已经完成工程链路补齐，但没有达到研究推进门槛。

可以确认：

- 真实传统 baseline 已接入。
- proposed 神经/RL 路线已接入 compare。
- checkpoint strict-load 可用。
- 在线 runner 不使用 Data 标签作为 reward。
- append-safe compare 可用。
- 全量测试通过。

不能声称：

- proposed 超过传统均衡器。
- RL 调制带来稳定增益。
- 当前 checkpoint 已训练完成。
- 当前结果可用于论文主结论。

## 6. 失败原因反思

核心失败不是 compare 或 baseline 公平性，而是 proposed 接收机尚未训练到可用区间：

1. 2-step smoke checkpoint 只验证链路，不可能达到 BER 目标。
2. 修复前的 checkpoint 训练对象是 identity 序列，不是 Level A/B 真实信道。
3. 修复后虽然数据流正确，但训练步数仍极小，BER 仍接近随机。
4. 传统 `SC-FDE-MMSE` 和 `DFE-RLS` 在当前 Level B smoke 配置下非常强，说明“传统方法一定差”的假设不能无条件成立。
5. Reward/Data Spearman 小样本为 NaN，说明当前 reward 设计还不能支撑 PPO 正式训练。

## 7. 下一阶段建议

不要直接跑正式矩阵。下一阶段应按以下顺序推进：

1. 先做 Level A 单配置离线训练，要求 `Offline NN only` 明显低于随机 BER。
2. 检查 `bit_error_rate()`、BPSK 符号、BCE target、logit 正负号是否完全一致。
3. 在 Level A 通过后，进入 Level B 单配置：

```text
delay=20, snr=10 或 15, pilot_total=128/160, multi_block
```

4. 训练到 Offline NN 在单配置低于传统弱 baseline，再谈 RL modulation。
5. 修复 Reward/Data Spearman NaN：需要增加 frame/seed/action 多样性，或重构 reward 为连续 loss 改善而非离散 BER 排序。
6. 只有 Offline NN 与 Reward/Data alignment 都可用后，再启动 pilot sweep。

## 8. 清理策略

本轮产生的 `logs/`、`pretrained/`、`__pycache__/` 属于本地运行产物，不纳入 git。保留 `.venv-gpu/`，因为这是用户要求使用 GPU 跑实验所需环境。
