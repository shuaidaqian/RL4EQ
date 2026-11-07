# block_phase 开发筛查：10 session × 64 帧 × 两种时序（2026-11-07）

状态：**开发机制筛查，不是确认。** 本文件回答两个问题：

1. 逐块 DD 相位（`block_phase`）相对当前 `soft` Data 后验，是否有可归因、可重复的 Data BER 增量？
2. 「同帧重解码」与「跨帧因果」两种合法时序的差异有多大，block_phase 的增量是否依赖非因果时序？

## 协议

- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`；强离线 checkpoint SHA256 `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4`。
- 新开发 seed：145001–145010（10 个独立 session；与确认 seed 100001–100010、既有开发 seed 90001–99010/130001–130003/140001–140002 隔离）。
- 每 seed/SNR 64 帧；四 SNR 全部报告，不按 SNR 删点。
- Frozen：同源 checkpoint + acquisition CIR + 同一 `dd_soft` tail。
- 变体：
  - `soft`：当前 Data 后验（`R_t = σ² + Σ|h_l|² Var_{t−l}`）；
  - `block_phase`：先用 Data soft 判决估计 4 个块的 DD 相位，再做 Data 后验。
- 两种时序（`--timing`）：
  - `causal`（可部署）：本帧判决只使用更新前的 CIR；本帧 Pilot/Data 后验更新只作用于下一帧，与已确认的 `diagnose_bayesian_cir.py` 一致。
  - `same_frame`（历史语义）：更新 CIR 后立刻重新解码本帧（同帧重解码 / transductive turbo）。合法但不属于已确认时序，不能直接部署。
- 公平性：同一 episode 内所有变体共享同一帧波形（`same_episode_for_all_variants=True`）；`soft`/`block_phase` 均不读 Data/Reward 标签（`non_oracle_uses_reward_or_data_labels=False`）。

## 结果：绝对 Bayes 增益（Frozen−Online BER）

### causal（可部署时序）

| SNR | Frozen | soft | block_phase | block−soft（95%） | block−soft 同向 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.198448 | 0.177922 | 0.174058 | +0.00386 `[−0.00418, +0.01191]` | 6/10 |
| 5 dB | 0.055428 | 0.049044 | 0.047622 | +0.00142 `[−0.00136, +0.00421]` | 6/10 |
| 10 dB | 0.007796 | 0.006285 | 0.006209 | +0.000075 `[−0.00004, +0.00019]` | 6/10 |
| 15 dB | 0.001147 | 0.001025 | 0.001005 | +0.000020 `[−0.00002, +0.00006]` | 5/10 |

配对增益（session 级，n=10，95% 描述性 t 区间）：

| SNR | soft | block_phase |
|---:|---:|---:|
| 0 dB | +0.02053 `[+0.00943, +0.03162]`，负向 1/10 | **+0.02439** `[+0.01636, +0.03242]`，负向 0/10 |
| 5 dB | +0.00638 `[−0.00001, +0.01278]`，负向 1/10 | **+0.00781** `[+0.00250, +0.01311]`，负向 0/10 |
| 10 dB | +0.00151 `[+0.00080, +0.00223]`，负向 0/10 | **+0.00159** `[+0.00083, +0.00234]`，负向 0/10 |
| 15 dB | +0.00012 `[−0.00004, +0.00028]`，负向 2/10 | +0.00014 `[−0.00003, +0.00031]`，负向 2/10 |

### same_frame（历史语义，非因果）

| SNR | Frozen | soft | block_phase | block−soft（95%） | block−soft 同向 |
|---:|---:|---:|---:|---:|---:|
| 0 dB | 0.198448 | 0.169293 | 0.169149 | +0.00014 `[−0.00676, +0.00704]` | 8/10 |
| 5 dB | 0.055428 | 0.046901 | 0.046562 | +0.00034 `[+0.00011, +0.00057]` | 8/10 |
| 10 dB | 0.007796 | 0.006169 | 0.006158 | +0.00001 `[−0.00006, +0.00008]` | 6/10 |
| 15 dB | 0.001147 | 0.001062 | 0.001056 | +0.000006 `[−0.00002, +0.00003]` | 6/10 |

配对增益：

| SNR | soft | block_phase |
|---:|---:|---:|
| 0 dB | +0.02915 `[+0.01821, +0.04010]`，负向 0/10 | +0.02930 `[+0.02330, +0.03530]`，负向 0/10 |
| 5 dB | +0.00853 `[+0.00308, +0.01397]`，负向 0/10 | +0.00887 `[+0.00340, +0.01434]`，负向 0/10 |
| 10 dB | +0.00163 `[+0.00089, +0.00236]`，负向 0/10 | +0.00164 `[+0.00089, +0.00238]`，负向 0/10 |
| 15 dB | +0.000085 `[−0.00008, +0.00026]`，负向 4/10 | +0.000092 `[−0.00007, +0.00025]`，负向 3/10 |

## 窗口形状（causal，16 帧窗口）

| SNR | soft 窗口 | block_phase 窗口 |
|---:|---|---|
| 0 dB | `+0.0052, +0.0165, +0.0231, +0.0373` | `+0.0096, +0.0253, +0.0222, +0.0404` |
| 5 dB | `+0.0033, +0.0050, +0.0108, +0.0064` | `+0.0026, +0.0054, +0.0115, +0.0117` |
| 10 dB | `+0.0009, +0.0013, +0.0018, +0.0021` | `+0.0010, +0.0014, +0.0018, +0.0021` |
| 15 dB | `+0.00005, +0.00007, +0.00005, +0.00032` | `+0.00006, +0.00007, +0.00008, +0.00036` |

0 dB 在 64 帧内仍呈「首窗正 → 后段更高 → 末窗为正」，与用户 2026-11-07 的 0 dB 判据（早期收敛/后段饱和+晚窗为正）方向一致；尚未到 128 帧的饱和段。

## 判断

1. **block_phase 相对 soft 没有可归因、可重复的增量。** causal 时序下四 SNR 的 `block_phase − soft` 配对 95% 区间全部跨零（0 dB `+0.0039 [−0.0042,+0.0119]`、5 dB `+0.0014 [−0.0014,+0.0042]`）；same_frame 下只有 5 dB 区间不跨零，但幅度仅 `+3.4e-4`，不具实际意义。
2. **没有升级为 0 dB 正式变体的证据。** 0 dB 的点估计 `+0.0039` 需要约 40+ 个 session 才能在 session 级检验中检出（按观测 sd≈0.011），远超当前可行预算；2-seed 初筛的 `+0.0018` 相对 soft 也未在 10 seed 上稳定放大。
3. **block_phase 也不劣于 soft。** 其绝对 Bayes 增益在 0/5/10 dB 均成立（10/10 seed 无负向、区间不跨零），NMSE 略低于 soft（causal 0 dB `0.0407` vs `0.0431`），可作为稳健性说明，不构成新机制。
4. **时序差异显著且方向符合预期。** same_frame 的绝对增益明显更高（0 dB `+0.0293` vs causal `+0.0244`；5 dB `+0.0089` vs `+0.0078`），因为同帧重解码额外使用了本帧后验信息；但它不是已确认协议的时序，不能直接部署或与确认结果混合报告。
5. **因果版本的绝对增益与确认 v1 方向一致。** 0/5/10 dB 的 block_phase 增益分别为 `+0.0244/+0.0078/+0.0016`，与确认协议的 `+0.0351/+0.0058/+0.0015`（10×128，不同 seed/帧数）同向；量级差异来自 64 帧与不同 session，不能直接比较。

**结论：不把 block_phase 升级为正式的 0 dB 变体，也不进入 P2 矩阵。** 保留为「与 soft 等价、无独立增量」的开发消融；若未来仍要检验，应使用更长会话与新的独立开发 seed，而不是在同一 10 seed 上追加。

## 成本

| 时序 | soft | block_phase | 说明 |
|---|---:|---:|---|
| causal | ≈660 ms/帧 | ≈665 ms/帧 | 不含同帧重解码；含 Pilot/Data 后验 |
| same_frame | ≈1270 ms/帧 | ≈1275 ms/帧 | 含更新后本帧重解码 |

block_phase 的相位估计相对 Data 后验约 +5 ms/帧；主要成本仍是模型前向与后验，详见[成本 profile](../profile_bayes_cost_v1_20261107/README.md)。两种时序均不满足 102.4 ms 帧预算。

## 复现

```powershell
$env:PYTHONUTF8=1

# causal（可部署时序）
python scripts/diagnose_0db_data_posterior.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 145001 145002 145003 145004 145005 145006 145007 145008 145009 145010 --frames 64 --variants soft block_phase --timing causal --device cuda --output docs/results/block_phase_dev10x64_v1_20261107/causal_10x64.json

# same_frame（历史语义对照）
python scripts/diagnose_0db_data_posterior.py --config configs/eme_long_memory_v2_online_dd_tail_v1.json --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt --seeds 145001 145002 145003 145004 145005 145006 145007 145008 145009 145010 --frames 64 --variants soft block_phase --timing same_frame --device cuda --output docs/results/block_phase_dev10x64_v1_20261107/same_frame_10x64.json

# 汇总
python scripts/analyze_block_phase_dev.py docs/results/block_phase_dev10x64_v1_20261107/causal_10x64.json docs/results/block_phase_dev10x64_v1_20261107/same_frame_10x64.json
```

## 边界

- 10 session × 64 帧开发筛查；不是确认，不能写成确认结论。
- 描述性 95% 区间使用 session 级 t 临界值（n=10 时 2.262），未做多重比较校正；仅作机制筛查。
- seed 145001–145010 一旦用于本筛查，不得再当作独立确认集。
- `same_frame` 是同帧重解码，不读标签但使用本帧后验更新后的 CIR，与已确认跨帧因果不同；报告中必须分开。
- 15 dB 错误数少（64 帧约 0.07 个错误/帧），配对差 1 个净比特即约 `1.5e-5`，不能据此判断 15 dB 的细微效应。
- 未评估 EM/`em_block` 变体在 10 seed 下的表现；本文件只回答 block_phase 问题。
