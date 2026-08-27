# EME-inspired slow-drift physical RL 研究重构方案

## 1. 目标重构

当前主目标调整为：

```text
在 EME-inspired Level B 慢时变长回波信道下，
先证明 Offline physical-aware NN 能超过公平传统均衡器，
再证明 Online physical RL 微调能进一步超过 Offline NN 和规则调制。
```

这一路线不再把 PPO 当作弥补弱接收机本体的补丁。若 Offline NN 本体明显弱于传统 baseline，则不扩大 PPO；若物理动作在 Reward Pilot 和 Data BER 上不可辨识，则不宣称 RL 有贡献。

## 2. 主场景定义

主场景固定为 Level B：

```text
delay = 20 / 30 / 40
SNR = 0 / 5 / 10 / 15 dB
pilot_total = 128
pilot_layout = prefix
```

EME-inspired impaired 机制采用：

- residual Doppler/CFO 作为主变量；
- slow phase drift 作为主变量；
- libration-like tap-gain drift 作为慢变复增益扰动；
- tap support 不做逐帧随机重采样；
- phase state 和 tap gain 状态跨帧连续相关；
- 所有在线 observation、reward、动作选择和调制更新不使用 Data 标签。

该场景不声称是完整物理 EME 仿真，而是用 EME 链路中的慢 Doppler、慢相位、libration fading/spreading 启发构造可控宽带长回波压力场景。

## 3. 公平性边界

所有方法共享 profile-level 先验：

```text
residual CFO budget
phase drift scale
tap drift rho
frame structure
pilot layout and total pilot count
```

所有方法都不能在线读取：

```text
真实 signed CFO
真实 phase state
真实 tap drift innovation
Reward Pilot 之外的在线标签
Data 标签或 BER_data 上界
```

传统 baseline 可使用 profile-level CFO budget 作为 pilot-based 搜索和限幅范围，但每帧 phase/CFO correction 必须只由接收信号和 Adapt Pilot 估计。Proposed 可使用相同先验做 phase/Doppler feature normalization 和安全动作边界。

## 4. traditional-only 工作区冻结

先不运行 Proposed。只运行 traditional + diagnostic 做工作区冻结。

扫描候选：

```text
cfo_abs = [0.0008, 0.0015, 0.0030, 0.0050]
phase_noise_std = [0.0008, 0.0015, 0.0030, 0.0050]
rho = [0.995, 0.990, 0.980]
```

共 `4 x 4 x 3 = 48` 个候选。每个候选运行：

```text
delays = 20 / 30 / 40
snrs = 0 / 5 / 10 / 15 dB
seeds = 2
frames = 20
pilot_total = 128
pilot_layout = prefix
methods = traditional + diagnostic
```

profile-aware traditional CFO limit：

```text
residual_cfo_limit = max(0.001, 1.5 * cfo_abs)
```

该 limit 是 profile-level 设计先验，不是 episode-level oracle。

## 5. 按 SNR 分层冻结目标

工作区冻结不要求所有 delay/SNR 配置落在同一 BER 区间，而按 SNR 分层判断最强 traditional baseline：

```text
0 dB:  best traditional BER_data 约 0.20-0.35
5 dB:  best traditional BER_data 约 0.12-0.28
10 dB: best traditional BER_data 约 0.08-0.22
15 dB: best traditional BER_data 约 0.05-0.18
```

候选排序只使用 traditional-only 结果，不看 Proposed。冻结后不得根据 NN 或 RL 结果调整 profile。

## 6. Proposed 结构方向

Offline 层：

```text
Physical-aware Neural Block Equalizer
```

要求：

- 使用 gated phase/Doppler correction branch；
- phase/Doppler feature normalization 使用共享 profile-level 先验；
- 与 unfolded equalizer 和 neural residual denoiser 结合；
- Offline NN only 必须先接近或超过最强 traditional baseline。

Online 层：

```text
Physical RL modulation
```

第一阶段只控制 phase/Doppler correction，不控制 tap-gain drift。RL 不直接输出 CFO、phase 轨迹或高维参数增量，只选择安全离散动作来控制：

- correction strength；
- phase state update rate；
- freeze/reset 策略。

第一版动作空间：

```text
0: keep
1: conservative
2: aggressive
3: freeze
4: reset
```

## 7. 成功门槛

严格门槛如下：

```text
Gate 1: 工作区冻结
  traditional-only 结果满足按 SNR 分层难度；
  冻结后不根据 proposed 结果改 profile。

Gate 2: Offline NN 超传统
  每个 SNR 层都优于最强 traditional baseline；
  mean BER 相对下降 >= 15%。

Gate 3: 动作可辨识
  phase/Doppler 动作在 Reward Pilot 上有稳定差异；
  Reward Pilot 改善与 Data BER 改善同向。

Gate 4: Online RL 有贡献
  RL phase-modulated NN 相对 Offline NN only 下降 >= 10%；
  相对 Rule phase modulation 下降 >= 5%；
  后半段 frames 优势更明显。

Gate 5: 统计成立
  seed/block bootstrap 95% CI 不跨 0。
```

## 8. C 阶段代码审查结论

从 `main` 新建分支后发现，`main` 本身仍是旧逐符号 A2C/PPO 项目形态，缺少 Level B、传统 baseline、compare、pytest 合约等当前研究线核心模块。因此当前分支显式引入 `codex/continual-ppo-unfolded-equalizer` 中的 Level B 研究骨架。

已确认当前骨架具备：

- `env/impairments.py`：支持 residual CFO 与 slow phase random walk；
- `env/extreme_delay_channel.py`：支持固定 support、跨帧 ISI、tap gain 慢漂移；
- `baseline/traditional_equalizers.py`：支持传统 baseline 与 pilot-based CFO/phase compensation；
- `evaluation/research_diagnostics.py`：已有 traditional-only Level B difficulty scan；
- `scripts/calibrate_traditional_difficulty.py`：已有命名 profile 校准入口；
- `agent/unfolded_equalizer.py`：已有 gated phase correction branch 的基础实现。

主要缺口：

- 现有校准入口只支持命名 impairment profile，不支持 48 候选数值化扫描；
- 传统 CFO limit 目前在 baseline 内部固定，尚不能随候选 profile 合理放宽；
- 缺少按 SNR 分层的 best-traditional 工作区评分和候选排序；
- 缺少 `eme_slow_drift_v1` 这类冻结 profile 输出。

下一步 B 阶段只实现 traditional-only 48 候选校准入口和评分输出，不训练 Proposed。

## 9. B 阶段 traditional-only 校准冻结结果

已完成 48 个候选的 traditional-only 扫描，输出位于：

```text
logs/eme_slow_drift_traditional_grid_2026-08-23/eme_slow_drift_grid.json
```

扫描共生成 `184320` 行逐帧结果。候选排序只使用传统非神经、非 RL baseline 的结果，没有读取 Proposed、Offline NN 或 RL 结果。

48 个候选中没有任何一个严格满足全部四个 SNR 分层目标。最佳候选为：

```text
candidate_id = cfo0p0008_phase0p003_rho0p995
cfo_abs_cycles_per_symbol = 0.0008
phase_noise_std = 0.003
rho = 0.995
residual_cfo_limit = 0.0012
acquisition_cfo_limit = 0.004
```

该候选的 best-traditional 分层结果为：

```text
0 dB:  CFO+DD-Phase DFE-RLS,       BER_data = 0.364648, 目标 0.20-0.35，略高 0.014648
5 dB:  CFO+DD-Phase LMMSE-FIR,     BER_data = 0.237956, 目标 0.12-0.28，通过
10 dB: CFO+DD-Phase DFE-RLS,       BER_data = 0.159375, 目标 0.08-0.22，通过
15 dB: CFO+DD-Phase LMMSE-FIR,     BER_data = 0.099935, 目标 0.05-0.18，通过
```

因此冻结 `eme_slow_drift_v1` 为 near-band candidate，而不是 strict all-SNR pass。后续所有 Offline/Online 实验均不得再根据 NN 或 RL 结果调整该 profile。

对应配置入口为：

```text
configs/continual_ppo_eme_slow_drift_v1.json
```

后续 Gate 解释调整为：

```text
Gate 1: 已接受 near-band freeze；严格 all-SNR pass 在 48 点网格内不存在。
Gate 2: 先训练 Offline NN，检验其是否能在该冻结 profile 上超过最强传统 baseline。
Gate 3-5: 只有 Gate 2 有实际优势后，才推进 Rule/Online RL 动作差异和统计显著性。
```
