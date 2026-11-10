# 0 dB 判据稳健性诊断（2026-11-10）

状态：**开发扩展筛查，不是正式确认。** 新 seed 170011–170020（10 session）；
不与已用开发/确认 seed 重叠。目的：检验 0 dB 的“早期收敛/后段饱和”判据是否跨 seed 稳健。

## 协议

- 配置：`configs/eme_long_memory_v2_online_dd_tail_v1.json`；`dd_soft` tail `alpha=1.0`。
- checkpoint：`D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`。
- seed：170011–170020；每 seed 128 帧；只跑 0 dB；`--timing causal`（本帧判决早于更新）。
- 变体：`soft`（当前 Data 后验）、`em2`（2 步 EM）、`em_block`（EM + 块相位）。
- 窗口：32 帧一个，共 4 窗；`gain = Frozen BER − 变体 BER`。
- Data 标签只事后 BER；真实 CIR 只事后 NMSE。

## 结果

### 窗口增益（均值 ± 95% CI，10 seed，t=2.262）

| 变体 | w1 | w2 | w3 | w4 | 晚窗 > 前 3 窗均值 |
|---|---:|---:|---:|---:|---:|
| soft | +0.0230 ± 0.0238 | +0.0235 ± 0.0269 | +0.0214 ± 0.0176 | +0.0486 ± 0.0222 | 8/10 seed |
| em2 | +0.0313 ± 0.0154 | +0.0154 ± 0.0177 | +0.0297 ± 0.0265 | +0.0383 ± 0.0268 | 5/10 seed |
| em_block | +0.0307 ± 0.0217 | +0.0274 ± 0.0238 | +0.0261 ± 0.0317 | +0.0520 ± 0.0217 | 6/10 seed |

### 绝对 BER（soft，均值 ± 95% CI）

| 窗口 | Frozen | Bayes |
|---|---:|---:|
| w1（0–31） | 0.2006 ± 0.0176 | 0.1777 ± 0.0144 |
| w2（32–63） | 0.2062 ± 0.0260 | 0.1826 ± 0.0107 |
| w3（64–95） | 0.2115 ± 0.0288 | 0.1901 ± 0.0251 |
| w4（96–127） | 0.2084 ± 0.0303 | 0.1598 ± 0.0165 |

## 结论

- **0 dB 的增益在四个窗口都存在**（soft 全部为正），但**不是单调下降/后段饱和**：
  - Frozen 的 BER 基本持平（0.201–0.212）；
  - Bayes 的 BER 在前三个窗口**略升**（0.178 → 0.183 → 0.190），第四个窗口才降到 0.160。
- 晚窗增益升高（0.049）主要来自 **Bayes 在 w4 变好**，不是 Frozen 恶化。
- **8/10 seed 的晚窗增益高于前 3 窗均值**，但 w1–w3 的 Bayes BER 并不单调下降。
- **旧“早期收敛/后段饱和”判据在新 seed 上不稳健**：它描述的是旧确认 seed（100001–100010）
  的形状；新 seed 呈现“波动 + 晚窗改善”，不是收敛/饱和。

## 建议

0 dB 判据建议改为：

1. **晚窗增益为正**（w4 的 `gain > 0`）；
2. **跨 seed 同向**（多数 seed 的晚窗增益为正）；
3. 不要求逐窗单调下降或后段饱和。

这保留了“0 dB 有可重复增益”的核心主张，同时不把“形状”钉在一个跨 seed 不稳定的模式上。
该修改需要用户确认后再写入路线文档与 AGENTS.md。

## 证据边界

- 新 seed 170011–170020 是开发 seed，**不是正式确认**；正式判据修改需要预注册。
- 旧确认 seed 100001–100010 的 0 dB 形状记录保留，不重测、不改写。
- `soft` 是当前实现；`em2`/`em_block` 仅作形状稳健性对照，不是主结果。
- 本诊断不改变在线只读约束。

## 复现

```powershell
$env:PYTHONUTF8=1
python scripts/diagnose_0db_data_posterior.py `
  --config configs/eme_long_memory_v2_online_dd_tail_v1.json `
  --checkpoint D:/Research/RL4EQ/pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --seeds 170011 170012 170013 170014 170015 170016 170017 170018 170019 170020 `
  --frames 128 --snrs 0 --timing causal --variants soft em2 em_block --device cuda `
  --output docs/results/0db_criteria_robustness_v1_20261110/variants.json
```

## 指纹

| 文件 | SHA256 |
| --- | --- |
| `model_best.pt` | `8cd3912d1c957c9689c51730aff9d27a8f74942186c802a7483e5781af65aec4` |
| `variants.json` | `d8e18793fe43d490e1979a0a48dddf8d6b665ffd8f2c1c386a955218fb729863` |
