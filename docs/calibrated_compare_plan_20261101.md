# 统一入口的传统校准协议与神经参考配对诊断

## 本阶段范围

将已经冻结的 FIR、DFE 和零填充 FDE 设置显式接入 `compare.py`，再测当前 head
checkpoint 与这些传统方法在同一轨迹上的差距。不调整神经结构或重训模型，
不根据本轮神经结果重新选择传统参数。额外线性块参考不进入主方法组。

统一入口新增 `--traditional-calibration configs/traditional_calibration_20261101.json`。
协议仅映射 `LMMSE-FIR`、`DFE-RLS`、`SC-FDE-MMSE`：前两者采用已冻结参数，
第三项显式采用去除估计历史后的零填充 Wiener 近似。结果记录算法版本、
协议 SHA256、同步方式和具体参数。同名历史结果不能与本协议混合。

协议包含校准来源配置和选择记录。执行前逐配置核验实际信道、时延、帧长、
Pilot/Reward 预算、扰动、相干时间和状态范围。神经模型结构不决定传统设置。
冻结协议只支持固定 acquisition CIR；不允许额外 CIR 更新。

## 实现门槛

- 统一入口在 4 个主 SNR 上精确复现接入前的存档核验轨迹，覆盖前 3 帧。
- 污染隐藏发送符号/标签不改变传统输出。
- 记录真实环境 seed 和接收波形 SHA256，每帧所有方法一致。
- 配置或 Pilot 不匹配、校准 seed 重用、不同协议 resume 时拒绝运行。
- 完整 pytest 及真实 checkpoint 的 GPU 比较 smoke。

## 预先固定的短诊断

- 配置：`configs/eme_long_memory_v2_head_transfer.json`，Level B、116、prefix
  Pilot=256、Reward=32、原 CFO/相位范围；不扩大失配。
- checkpoint：本阶段之前已经固定的
  `pretrained/head_transfer_post256_20261101/model_best.pt`，不使用新测试选模型。
- 方法：冻结协议的三类传统方法、`Frozen Offline NN`、正式
  `RL-Modulated Neural Block Equalizer`；沿用 head 动作集合，不新增动作。
- heldout_edge、CLI seed=47001–47005（统一入口实际环境 seed=97001–97005），
  0/5/10/15 dB、每条 8 帧，窗口/更新间隔均为 8。共 800 条逐帧记录。
- 8 帧只用于同轨迹协议和当前差距诊断，不能证明长期 PPO 学习效果；
  相对 Frozen 的增益和传统对照差距分开报告，不把辅助 BER<0.01 当作第一门槛。

独立 seed 的配对统计先在轨迹内求均值；不把重复帧当作独立样本。
本轮完整配置原本已声明 diagnostic_only，不进入正式长期主平均。

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe compare.py --config configs/eme_long_memory_v2_head_transfer.json --traditional-calibration configs/traditional_calibration_20261101.json --methods 'LMMSE-FIR' 'DFE-RLS' 'SC-FDE-MMSE' 'Frozen Offline NN' 'RL-Modulated Neural Block Equalizer' --pretrained pretrained/head_transfer_post256_20261101/model_best.pt --snrs 0 5 10 15 --seed-start 47001 --num-seeds 5 --frames 8 --state-split heldout_edge --update-interval 8 --device cuda --output-dir logs/calibrated_compare_head256_5s8f_20261101
```
