# 在线 PEFT 矩阵自动汇总

该表由 `scripts/summarize_online_matrix.py` 从 `frame_metrics.jsonl` 生成。
改善值定义为 `(Frozen BER - Online BER) × 100`，单位是百分点。

## SNR 汇总

| 日志目录 | SNR | 正收益 seed | 平均改善 (pp) | 最差 seed 改善 (pp) | Frozen BER | Online BER | 在线使用 Data 标签 |
|---|---:|---:|---:|---:|---:|---:|---|
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 5.0 | 5/5 | 0.0705 | 0.0109 | 4.8155% | 4.7450% | 否 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 10.0 | 2/5 | 0.0033 | 0.0000 | 0.6999% | 0.6966% | 否 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 15.0 | 4/5 | 0.0260 | 0.0000 | 0.1063% | 0.0803% | 否 |

## Seed 配对明细

| 日志目录 | SNR | seed | Frozen BER | Online BER | 改善 (pp) | 接受更新 | 回滚 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 5.0 | 72 | 4.5030% | 4.4813% | +0.0217 | 2 | 2 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 5.0 | 73 | 6.2283% | 6.2012% | +0.0271 | 1 | 1 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 5.0 | 74 | 3.0328% | 2.8158% | +0.2170 | 7 | 5 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 5.0 | 75 | 4.8665% | 4.8557% | +0.0109 | 2 | 2 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 5.0 | 76 | 5.4470% | 5.3711% | +0.0760 | 1 | 1 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 10.0 | 72 | 0.6565% | 0.6565% | +0.0000 | 1 | 1 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 10.0 | 73 | 1.2967% | 1.2967% | +0.0000 | 1 | 1 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 10.0 | 74 | 0.1845% | 0.1790% | +0.0054 | 4 | 3 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 10.0 | 75 | 0.6510% | 0.6402% | +0.0109 | 2 | 2 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 10.0 | 76 | 0.7107% | 0.7107% | +0.0000 | 0 | 0 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 15.0 | 72 | 0.1139% | 0.1085% | +0.0054 | 4 | 0 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 15.0 | 73 | 0.2387% | 0.1302% | +0.1085 | 7 | 0 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 15.0 | 74 | 0.0326% | 0.0326% | +0.0000 | 4 | 0 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 15.0 | 75 | 0.0814% | 0.0705% | +0.0109 | 4 | 0 |
| `formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f` | 15.0 | 76 | 0.0651% | 0.0597% | +0.0054 | 1 | 0 |
