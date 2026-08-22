# 迭代记录：phase vector conditioner 接入与初筛

时间戳：2026-08-12 16:05:30  
分支：`codex/continual-ppo-unfolded-equalizer`  
工作目录：`D:\Research\RL4EQ`

## 1. 本轮目标

本轮继续沿着“先增强神经接收机本体，再决定是否扩大 RL”的路线推进。具体目标是把 Adapt Pilot 前缀切分为多个局部子窗口，从每个窗口提取更丰富的相位残差信息，替代之前只输入 `phase_residual` 和 `cfo_residual` 两个标量的弱 conditioner。

核心假设：

- 如果传统 baseline 在轻度相位扰动下仍然很强，那么神经网络需要更显式地获得接收端可见的相位/残差信息。
- 只给两个全局标量可能不足以描述整帧内的局部相位漂移与 residual 分布。
- richer phase vector 如果有效，应首先改善 `Offline NN only`，而不是直接期待 PPO 扭转接收机本体不足。

## 2. 实现内容

### 2.1 新增局部 phase residual vector

已在 `baseline/traditional_equalizers.py` 中实现：

```python
estimate_phase_residual_vector(receiver_view, cir, soft_tail, blocks=4)
```

该函数只使用在线接收端允许可见的信息：

- 接收帧 `rx_symbols`
- Adapt Pilot mask 与 Adapt Pilot 已知符号
- 当前接收机维护的 CIR
- 当前接收机 soft tail

不读取：

- Reward Pilot 标签
- Data 标签
- 真实 impairment 参数
- true CIR

默认 `blocks=4`，输出 16 维向量。每个 block 输出 4 个统计量：

1. 局部 phase intercept
2. 局部 phase slope / CFO cycles per symbol
3. 局部 phase residual variance
4. normalized residual energy

### 2.2 `condition_from_cir()` 支持 rich feature vector

已修改 `agent/cir_estimator.py`：

```python
condition_from_cir(
    cir,
    snr_db,
    phase_residual=0.0,
    cfo_residual=0.0,
    phase_features=None,
)
```

兼容旧接口：

- 不传 `phase_features` 时，仍将旧的两个标量写入 latent 前两维。
- 传入 `phase_features` 时，将 feature vector 写入 `latent_residual` 前若干维，本轮默认前 16 维有效。

这样不会破坏已有旧测试和旧调用方式。

### 2.3 离线训练路径接入 vector conditioner

已修改 `training/curriculum.py`：

- 训练 batch 中每个样本都调用 `estimate_phase_residual_vector(..., blocks=4)`。
- `latent_residual` 由 `_phase_feature_latent()` 生成，前 16 维为 phase vector。
- `_validate_offline_nn()` 同步使用 phase vector，避免训练/验证条件分布不一致。

### 2.4 在线与比较路径接入 vector conditioner

已修改：

- `compare.py`
- `training/windowed_discrete_ppo.py`
- `training/rl_modulated_online.py`

现在 proposed / RL 路径都使用：

```python
phase_features = estimate_phase_residual_vector(...)
condition = condition_from_cir(..., phase_features=phase_features)
```

这保证预训练、在线 runner、正式 compare 的 condition 语义一致。

## 3. 测试与验证

### 3.1 TDD RED

新增测试：

```python
test_condition_from_cir_accepts_rich_phase_feature_vector
```

初次运行失败符合预期：

```text
TypeError: condition_from_cir() got an unexpected keyword argument 'phase_features'
```

该失败证明测试确实覆盖了新增接口，而不是测试已有行为。

### 3.2 单点 GREEN 测试

命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_traditional_baselines.py::test_phase_residual_vector_splits_adapt_pilot_into_local_statistics tests/test_receiver_architecture.py::test_condition_from_cir_accepts_rich_phase_feature_vector -q -p no:cacheprovider
```

结果：

```text
2 passed in 3.95s
```

### 3.3 相关模块测试

命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_traditional_baselines.py tests/test_receiver_architecture.py tests/test_evaluation_contract.py::test_compare_cli_writes_real_level_b_metrics -q -p no:cacheprovider
```

结果：

```text
23 passed in 33.42s
```

### 3.4 全量测试

第一次用 300 秒 timeout 执行全量测试时超时，未输出失败断言。随后按文件分组定位，发现主要是测试总耗时较长，尤其 `test_meta_adaptation.py`，不是功能断言失败。

最终使用更长 timeout 重新执行：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

结果：

```text
156 passed in 297.75s (0:04:57)
```

结论：当前代码测试健康，但全量测试耗时接近 5 分钟，后续若继续增加真实实验型测试，需要注意测试分层，避免默认单元测试越来越慢。

## 4. Smoke 与初筛实验

### 4.1 phase-vector smoke 预训练

命令：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo_phase_tiny.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/phase_tiny_phasevector_smoke_2026-08-12
```

结果：

```text
saved pretrained/phase_tiny_phasevector_smoke_2026-08-12
```

### 4.2 phase-vector smoke compare

命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo_phase_tiny.json --method-group proposed --pretrained pretrained/phase_tiny_phasevector_smoke_2026-08-12/model_best.pt --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 128 --pilot-layout prefix --impairment-profile phase_tiny --resume --output-dir logs/compare_phasevector_smoke_2026-08-12
```

结果：

```text
saved logs/compare_phasevector_smoke_2026-08-12
```

### 4.3 phase-vector 100-step 初筛预训练

命令：

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo_phase_tiny.json --stage all --steps 100 --batch-size 2 --amp --save-dir pretrained/phase_tiny_phasevector_100steps_2026-08-12
```

结果：

```text
saved pretrained/phase_tiny_phasevector_100steps_2026-08-12
```

训练指标摘要：

- `condition_cir_sources`：
  - true: 200
  - acquisition: 200
  - noisy: 200
  - dd_like: 200
- `offline_nn_validation.mean_ber_data`: `0.013139`
- `offline_nn_validation.gate_pass`: `true`
- `best_model_metric`: `0.013139`

注意：这里的 offline validation 只是轻量验证，不等价于正式 Level B 主矩阵。

### 4.4 phase-vector 100-step 小矩阵

命令：

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo_phase_tiny.json --method-group all --pretrained pretrained/phase_tiny_phasevector_100steps_2026-08-12/model_best.pt --delays 20 --snrs 10 15 --num-seeds 1 --frames 4 --pilot-total 128 --pilot-layout prefix --impairment-profile phase_tiny --resume --output-dir logs/compare_phasevector_100steps_2026-08-12
```

结果文件：

- `logs/compare_phasevector_100steps_2026-08-12/frame_metrics.jsonl`
- `logs/compare_phasevector_100steps_2026-08-12/summary.json`

方法均值：

| 方法 | mean BER_data | 样本数 |
|---|---:|---:|
| DFE-RLS | 0.000000 | 8 |
| Perfect-CSI Block | 0.000000 | 8 |
| Fixed CG-BPSK-DD Block Detector | 0.000651 | 8 |
| SC-FDE-MMSE | 0.001953 | 8 |
| Offline NN only | 0.028646 | 8 |
| NN + Fixed Modulation | 0.028646 | 8 |
| NN + Discrete PEFT Scheduler | 0.028646 | 8 |
| RL-Modulated Neural Block Equalizer | 0.028646 | 8 |
| RLS Linear | 0.033203 | 8 |
| NN + Rule Modulation | 0.035156 | 8 |
| LMMSE-FIR | 0.036458 | 8 |
| LMS | 0.184245 | 8 |
| NLMS | 0.296549 | 8 |

CFO-corrected / DD-phase corrected variants在本次 `phase_tiny` 小矩阵中出现局部异常劣化，说明补偿器不是当前主结论依据；主比较仍应关注 DFE-RLS、SC-FDE-MMSE、LMMSE/RLS Linear 与 Proposed。

## 5. 当前效果判断

本轮代码层面目标已经达成：phase vector conditioner 已经贯通训练、在线、比较三条路径，并通过全量测试。

但从 100-step 初筛结果看，效果没有达到预期：

- 当前小矩阵中 DFE-RLS 已经达到 `0 BER_data`。
- SC-FDE-MMSE 也只有 `0.001953`。
- `Offline NN only` 为 `0.028646`，仍明显弱于 DFE-RLS 和 SC-FDE-MMSE。
- RL / PEFT 类方法在本次短训练 checkpoint 上没有改善 Offline NN。

因此，本轮结果不支持马上扩大 PPO 训练。原因不是 RL 不需要，而是接收机本体还没有站到足够强的位置；在接收机本体弱于传统强 baseline 的情况下，PPO 主要是在弱接收机附近做动作搜索，很难稳定产生超过传统的结果。

## 6. 对当前研究假设的反思

### 6.1 “极端长时延传统一定很差”这个判断需要更精确

实验继续说明：只要信道是线性、BPSK、无编码、prefix pilot 足够长、接收机允许整帧处理，传统 DFE-RLS / SC-FDE-MMSE 并不会天然失败。长时延本身不必然使传统均衡器崩溃；真正会拉开差距的是：

- 估计信息不足；
- 非平稳相位/频偏导致单帧估计跨帧失配；
- 支撑集或 tap 增益变化使传统递推适配滞后；
- 非线性或模型失配；
- 或者传统算法复杂度被合理约束。

当前用户明确暂时不加入 CFO、额外相位扰动、非线性、编码、高阶调制，因此在 clean/phase_tiny 的高 SNR 子区间，传统 baseline 强是合理结果。

### 6.2 richer conditioner 不是充分条件

phase vector 比两个标量更合理，但它只是把诊断特征输入神经网络；如果主干检测器/训练目标仍不足以利用这些特征，它不会自动超过 DFE。

当前 100-step 初筛更像是在验证“链路可跑通”，还不能说明 phase vector 本身无效。要判断是否有效，需要更长训练或更聚焦于 DFE 尚未饱和的配置，例如低 SNR、delay 30/40、以及更困难但仍合理的 Level B 子区间。

### 6.3 PPO 不应在当前节点扩大

PPO 的目标是在线根据 Adapt/Reward Pilot 反馈选择安全动作，使 Proposed 随帧持续变好。但如果 Offline NN only 明显弱于传统，PPO 的 reward 只能在弱接收机局部扰动中寻找小收益，不太可能补足架构差距。

当前更合理的门槛仍是：

```text
先让 Offline NN / Rule modulation 接近或超过传统强 baseline，
再扩大 Continual PPO。
```

## 7. 下一步建议

下一阶段不建议继续盲目训练 PPO。建议按以下顺序推进：

1. 在 `phase_tiny` 下跑更覆盖主矩阵的 phase-vector checkpoint，对比旧 phase-aware 500-step，确认 vector 是否在 d30/d40 和 SNR 0/5/10/15 上有稳定收益。
2. 如果收益有限，优先做显式 phase-correction branch：让神经接收机内部有一条可学习/可控的相位补偿路径，而不是只把相位统计塞进 latent。
3. 对 `NN + Rule Modulation` 做 behavior cloning 初始化或规则蒸馏，因为之前多轮结果显示 rule modulation 经常比 PPO 更稳，说明有效动作方向存在，但 PPO 未学会。
4. 继续保持 traditional baseline 非神经、非 RL，但不要再用高 SNR、d20、phase_tiny 这种 DFE 已经 0 BER 的子集作为主要判断区间。
5. 只有当 Offline NN / Rule modulation 在主工作区间接近 DFE-RLS 或 SC-FDE-MMSE 后，再启动更长的 Continual PPO 训练。

## 8. 本轮清理说明

本轮没有删除实验输出目录：

- `pretrained/phase_tiny_phasevector_smoke_2026-08-12`
- `pretrained/phase_tiny_phasevector_100steps_2026-08-12`
- `logs/compare_phasevector_smoke_2026-08-12`
- `logs/compare_phasevector_100steps_2026-08-12`

原因：这些是本轮结论的证据工件，且已经被 `.gitignore` 忽略，不会进入代码提交。

本轮检查到旧两标量函数 `estimate_phase_residual_features()` 仍保留。它不是主链路依赖，但用于兼容旧接口测试，因此暂不删除。

