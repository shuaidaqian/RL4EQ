> 状态：历史诊断/已被当前主路线取代。原始结果保留用于追溯，不代表当前默认方法。
> 当前规范：docs/state_tracking_receiver_design.md；验证：docs/state_tracking_validation_protocol.md。

# Reward logits 守门复核（2026-10-15）

## 目的

本轮复核固定离线 checkpoint 和在线微调结构，只检查 Reward Pilot 上的
`logits` 改变量上限 `0.1` 能否减少 Pilot 到 Data 的方向失配。该守门只使用
Reward Pilot，不读取 Data 标签；Data BER 只在运行结束后做配对评估。

## 固定条件

- checkpoint：`pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt`
- checkpoint SHA256：`8CD3912D1C957C9689C51730AFF9D27A8F74942186C802A7483E5781AF65AEC4`
- 信道：Level B、`eme_long_memory_v2`、`delay=116`
- 场景：`gap=0`，`cfo_phase_tiny`，coherence time 为 1200 s
- 帧结构：prefix Pilot，总长 256，其中 Adapt 224、Reward 32，Data 768
- 在线对象：均衡器内部 `phase_trend` Adapter；离线主干和 checkpoint 不变
- Adapt 目标：Pilot 复数信号重构，AdamW，5 步，学习率 0.005，参数增量范数上限 0.05
- Reward 守门：双窗口信号重构、Reward 硬错数不增加、Reward logits 平均改变量不超过 0.1
- 调度：fixed、每帧尝试；SNR 低于 5 dB 冻结
- 新验证 seed：72--76，每个 seed 24 帧
- 在线标签隔离：所有记录 `data_labels_used_online=false`

复现命令：

```powershell
.\.venv-gpu\Scripts\python.exe -B compare.py `
  --config configs/online_phase_signal_state_embedding_formal_moderate.json `
  --method-group proposed `
  --pretrained pretrained/eme_bce_all_32_20260905_pilot256/model_best.pt `
  --delays 116 --snrs 5 10 15 --num-seeds 5 --seed-start 72 --frames 24 `
  --pilot-total 256 --pilot-layout prefix --scheduler fixed --cir-update fixed `
  --update-interval 1 --online-condition-source pilot_cir_phase `
  --online-optimizer adamw --online-objective pilot_signal_reconstruction `
  --online-learning-rate 0.005 --online-steps 5 --online-max-delta-norm 0.05 `
  --online-proximal-weight 0.01 --online-min-reward-improvement 0.0005 `
  --online-reward-windows 2 --online-require-reward-signal-reconstruction `
  --online-reward-signal-tolerance 0 --online-cross-frame-tolerance 0.00001 `
  --online-reward-logit-change-tolerance 0.1 --online-freeze-below-snr-db 5 `
  --output-dir logs/formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f `
  --resume
```

## 结果

改善定义为 `(Frozen BER - Online BER) * 100`，单位为百分点。每一行是 5 个
独立 seed 的 24 帧平均，再按 seed 配对计算。

| SNR | Frozen BER | Online BER | 平均改善 | 正收益 seed | 最差 seed | 接受更新总数 | 回滚总数 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 dB | 4.8155% | 4.7450% | +0.0705 pp | 5/5 | +0.0109 pp | 13 | 11 |
| 10 dB | 0.6999% | 0.6966% | +0.0033 pp | 2/5 | 0 pp | 8 | 7 |
| 15 dB | 0.1063% | 0.0803% | +0.0260 pp | 4/5 | 0 pp | 20 | 0 |

逐 seed 配对结果：

| SNR | seed 72 | seed 73 | seed 74 | seed 75 | seed 76 |
|---:|---:|---:|---:|---:|---:|
| 5 dB | +0.0217 | +0.0271 | +0.2170 | +0.0109 | +0.0760 |
| 10 dB | 0 | 0 | +0.0054 | +0.0109 | 0 |
| 15 dB | +0.0054 | +0.1085 | 0 | +0.0109 | +0.0054 |

## 判断

1. `0.1` 守门没有让 Adapter 失去作用。5 dB 和 15 dB 新 seed 全部非负，说明
   受限 PEFT 更新可以把部分收益迁移到 Data 段；接受后发生的回滚也确实减少了
   一部分风险。
2. 10 dB 仍只有两个 seed 有正收益，平均改善接近零。它说明 Reward Pilot 的
   连续指标仍不能稳定排序 Data BER，不能把该守门写成通用解决方案。
3. 结合此前 seed 57--61 的 `0.1` 复核，结果仍是“平均有小收益、跨 SNR 稳定性
   不足”，不能宣称在线阶段已经在每个 SNR 上稳定明显超过离线训练。
4. 这轮没有改离线训练、checkpoint、Pilot 布局或主配置，也没有引入 RL/Bandit。
   当前方法仍是 Adapt Pilot 更新内部 Adapter，Reward Pilot 验收和回滚，Data
   标签只做事后评估。

原始逐帧日志：
`logs/formal_signal_guard_logitcap01_gap0_5s_5_10_15db_seed72_76_24f/`

自动汇总：
`scripts/summarize_online_matrix.py`

当前结论是：方法链路已经闭环，PEFT 更新能够改变 Data 输出并在部分 SNR 带来收益；
尚未达到“所有主 SNR、多个 seed 都稳定且明显超过 Frozen Offline NN”的论文成功门槛。
