# DFE-RLS 先验与遗忘因子的独立校准

上一轮只比较 41+5 与 117+116 两种 DFE 记忆长度，遗忘因子和逆协方差先验
写死为 0.995 和 10，不能据此声称长 DFE 已充分调优。本阶段只使用传统结果
选择参数，不加载神经模型，不修改信道、同步、Pilot 或正式主方法集合。

## 看新输出前固定的网格与预算

- 配置：`configs/eme_long_memory_v2_head_transfer.json`，与首轮传统校准相同。
- 原 acquisition CFO 先验加当前 Adapt 公共相位补偿，各候选保留自身历史 tail。
- 前馈/反馈：41+5、117+116。
- 遗忘因子：0.98、0.995、1.0。
- 初始逆协方差尺度：0.001、0.01、0.1、1、10、100；对应初始精度是其倒数。
- 共 36 个设置，含原 41+5、0.995、10 参考。
- offline_train，新 seed=40000–40002、每条 4 帧、四主 SNR；共 1728 条候选帧记录。
- 每 SNR 按校准 seed 平均 Data BER 选择；并列优先较短记忆、较少遗忘、较强先验。
- 看留出数据前冻结设置及校准文件 SHA256。预定 heldout_edge 新 seed=59001–59005、
  每条 8 帧、四 SNR，仅核验所选设置和原参考；不在留出集重新扫网格或选参数。
- 参数每帧按现有 DFE 规则重置；各自的历史判决 tail 递推保留。Data 标签只在
  当帧所有估计及递推结束后用于离线评分，Reward 标签不参与训练。

本轮扩展 `_dfe_rls` 的可选参数，默认值保持原行为。反馈特征用等价切片替代
逐标量堆叠；推理中已处理位置本就是硬判决，移除反复对整帧作硬转换。
独立加权批量岭回归解与既有真实轨迹 fixture 必须通过，才执行完整校准。

首次完整预算尝试在两个 seed 完成后，因发现协方差更新含不必要的三次方矩阵
乘法而停止；没有导出校准或冻结文件，没有读取逐配置 BER 选参。改为代数等价的
秩一更新括号顺序后，独立加权批量解和既有轨迹的 40 项定向回归通过，再按同一
网格、seed 和预算从头执行。该中止仅处理计算成本，不改变协议或增加独立样本。

## 判读

核验以 seed 为单位计算所选设置相对原参考的配对区间。若训练范围更好但留出
不稳定，保留这个失败结果，不把训练选优当作泛化收益。无论结果如何，不能
据 36 个设置推出传统 DFE 所有超参数已经全局最优。

首次实际信道校准之前，参照上轮传统 FIR 校准中较强正则的结果，在最初四个
先验尺度之外补入 0.001/0.01，避免先验网格只覆盖弱正则。此前仅运行合成
批量解与 mock 协议测试，尚未查看本轮真实信道校准输出；完整运行按上述 36 项冻结。

本阶段是独立校准诊断，尚不自动覆盖现有冻结传统协议；若需要接入统一 compare，
必须显式记录新协议、参数、来源与 seed 排除，并保留旧协议的可复现性。

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/calibrate_dfe_rls.py --config configs/eme_long_memory_v2_head_transfer.json --seed-start 40000 --num-seeds 3 --frames 4 --freeze-output logs/dfe_rls_frozen_selection_20261101.json --output logs/dfe_rls_calibration_3s4f_20261101.json
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/calibrate_dfe_rls.py --config configs/eme_long_memory_v2_head_transfer.json --state-split heldout_edge --selection logs/dfe_rls_frozen_selection_20261101.json --seed-start 59001 --num-seeds 5 --frames 8 --output logs/dfe_rls_confirmation_5s8f_20261101.json
```
