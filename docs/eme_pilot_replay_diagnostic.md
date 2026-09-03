# EME Pilot-only Replay 诊断结果

## 诊断目的

本诊断使用已经完成的 `compare.py` 逐帧日志，离线重建每个真实调度帧对应的保持窗口，检验当前 Reward Pilot 损失改善能否预测后续 Data BER 改善。Data 标签只在本报告的事后诊断中使用，不进入在线 observation、动作选择、梯度更新或回滚规则。

该诊断不是新的在线算法，也不把 Online 与 Frozen 的实际差异重新伪装成在线 Reward。它只回答一个进入在线 Bandit 或窗口验收前必须回答的问题：当前 Pilot-only 信号是否具有足够的排序能力。

## 输入与协议

输入目录：

```text
logs/eme_update_rate_main_30f_3s/
```

该目录包含 Level B、`eme_long_memory_v2`、`max_delay=116`、总 prefix Pilot=128、Adapt 96 / Reward 32、`update_interval=8`、3 seeds、0/5/10/15 dB、30 帧的 720 条逐帧记录。

replay 事件只从 `online_update_scheduled=true` 的帧开始，使用该事件开始的连续保持窗口。报告同时保存：

- `reward_loss_improvement`：调度帧 Reward Pilot BCE 的前后改善；
- `data_ber_improvement`：窗口内 Frozen BER 减 Online BER；
- `future_data_ber_improvement`：排除事件帧后的后续窗口 Data BER 改善；
- `peft_update_applied`：该调度事件是否实际接受 PEFT。

运行命令：

```powershell
.\.venv-gpu\Scripts\python.exe scripts/diagnose_pilot_replay.py `
  --log-dir logs/eme_update_rate_main_30f_3s `
  --output-dir logs/eme_update_rate_main_30f_3s/replay_diagnostic `
  --window-sizes 1 2 4 8
```

## 结果

窗口长度为 1 时没有后续帧，因此不能计算 `future_data_ber_improvement`。窗口 2/4/8 的所有调度事件汇总如下：

| 窗口 | 事件数 | 可用事件数 | Reward 与后续 Data 的 Spearman | 后续 Data 平均改善 |
|---:|---:|---:|---:|---:|
| 2 | 48 | 48 | 0.037 | -0.346 pp |
| 4 | 48 | 48 | -0.227 | -0.318 pp |
| 8 | 48 | 48 | -0.151 | +0.033 pp |

分 SNR 结果：

| 窗口 | 0 dB | 5 dB | 10 dB | 15 dB |
|---:|---:|---:|---:|---:|
| 2 | NaN | 0.397 | 0.021 | -0.014 |
| 4 | NaN | 0.393 | -0.304 | -0.616 |
| 8 | NaN | 0.480 | -0.171 | -0.591 |

0 dB 的 PEFT 按协议冻结，因此 Reward loss 改善没有变化，相关性不定义。5 dB 的相关性最好也只有 0.480，低于预设的 0.6 门槛；10/15 dB 在窗口 4/8 显著转为负值。

只看已经接受 PEFT 的 13 个事件时，窗口 2/4/8 的全体相关性分别为 `-0.116/-0.104/-0.170`。这说明“Reward Pilot 上接受”并不等价于“对后续 Data 有益”，而且简单窗口平均没有消除失配。

## 研究判断

当前不能把 Contextual Bandit 接入主实验。Bandit 可以改变动作选择器，但不能修复一个不能稳定排序候选好坏的 Reward 代理；在这个阶段接入 Bandit 会把 Reward 失配隐藏在策略学习噪声中。

当前主要矛盾已经从 Pilot 资源数量收敛为两点：

1. `phase/CIR` 估计得到的状态没有充分表达长记忆信道对均衡器参数的影响；
2. 当前 PEFT 的 Adapt Pilot 梯度方向与后续 Data 段的目标不一致，单帧或短窗口 Reward 都无法可靠验收。

下一步应优先做受控的在线状态表示实验：保持 Level B 信道、总 Pilot 和 checkpoint 不变，比较只更新低维物理校正状态、只更新 head、以及当前 adapter/FiLM 组合，并用相同保持窗口进行 replay。候选动作仍然只允许改变受限校正强度和保持帧数；只有某一类状态/参数更新在多个 SNR 上通过 Pilot-only 排序门槛，才继续实现 Bandit。

## 输出文件

- `scripts/diagnose_pilot_replay.py`
- `evaluation/research_diagnostics.py` 中的 `build_pilot_replay_events`
- `logs/eme_update_rate_main_30f_3s/replay_diagnostic/summary.json`
- `logs/eme_update_rate_main_30f_3s/replay_diagnostic/replay_events.jsonl`
