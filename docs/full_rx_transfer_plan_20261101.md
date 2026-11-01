# 整帧接收输入的匹配离线训练与四臂机制验证

用户已明确允许均衡器在在线更新与验收时读取整帧已接收波形，包括 Data 区。
更新损失仅用 Adapt 标签；动作后的验收与 reward 仅用 Reward 标签；未知发送
符号与 Data 标签不进入在线控制。这项答复解除输入协议选择的阻碍。
神经调制结构的另一项选择尚未答复，本轮保持已有网络与 phase 参数组。

此前只改变输入范围的 48 上下文组件实验没有证明可靠收益，但源 checkpoint
的历史内循环仍使用置零输入。本轮检验与完整接收输入匹配的离线训练，不重复
扩大此前不匹配配置的测试矩阵，不声称更换输入必然成功。

## 查看本轮结果前固定

- 源为 `pretrained/phase_transfer_post256_20261101/model_best.pt`，SHA256：
  `305ea6f636cf5f0619649489a7d2e5ed5ce9138334e37e731bf24246a78fe010`。
  继承其全部历史训练开销；A/B 均从此源重新开始外层优化器。
- 配置为 `configs/eme_long_memory_v2_full_rx_transfer.json`，不改信道、扰动、
  前缀 Pilot、acquisition、tail 规则、网络或 RL 动作。
- 显式设置 `inner_receiver_input=full_rx`、`reward_receiver_input=full_rx`。
  仍使用 phase 组、Adapt BCE、AdamW lr=5e-4、一步、半径 0.05。
- A：BCE_all；B：BCE_all + 更新后 Data BCE，权重均为 1。
- 两分支各 256 步、batch=1、外层 lr=1e-4，offline_train seed=44000–44255，
  四主 SNR 轮换。FFT、float32、math attention，不使用 AMP。
- GPU 两步 smoke 只核验执行能力，正式 A/B 不从 smoke 权重接续。
- 首次四臂评估：heldout_edge seed=60001–60005，各 8 帧，0/5/10/15 dB，
  每帧参数重置。两个分支共享接收波形，每个分支有自身同 checkpoint Frozen。
- checkpoint 固定为预算终点，不用测试集挑选。相同训练 seed/SNR、源 SHA、
  信道状态与训练/评估输入协议都需核验。

## 统计与退出条件

同时检查 B Online 相对 B Frozen 的改善、B Online 相对 A Online 的改善，以及
B 与 A 的在线增益差。每个 SNR 单独报告 seed 级区间，不混合 SNR。
不能用 B Frozen 退化导致的更大适配幅度宣称成功。

若 B 出现可靠额外收益，再加入不含内循环的同权重 Data 重加权对照与独立确认；
单次迁移通过后才进入累计适配和正式离散安全 RL 窗口调度。
若仍未成立，不扩大同配置留出矩阵，转向候选作用方向与反馈可分辨性的分析。
15 dB Reward 饱和仍可能阻碍验收，不能为提高接受率事后放松门槛。

该阶段是机制实验，不进入主平均，不替代四 SNR 超过校准传统基线、独立在线
收益及后期优势更强的最终要求。

## 复现

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe pretrain.py --config configs/eme_long_memory_v2_full_rx_transfer.json --stage pilot_transfer --transfer-objective post_adapt --resume pretrained/phase_transfer_post256_20261101/model_best.pt --steps 2 --batch-size 1 --device cuda --save-dir pretrained/full_rx_transfer_smoke_20261101
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe pretrain.py --config configs/eme_long_memory_v2_full_rx_transfer.json --stage pilot_transfer --transfer-objective baseline --resume pretrained/phase_transfer_post256_20261101/model_best.pt --steps 256 --batch-size 1 --device cuda --save-dir pretrained/full_rx_transfer_baseline256_20261101
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe pretrain.py --config configs/eme_long_memory_v2_full_rx_transfer.json --stage pilot_transfer --transfer-objective post_adapt --resume pretrained/phase_transfer_post256_20261101/model_best.pt --steps 256 --batch-size 1 --device cuda --save-dir pretrained/full_rx_transfer_post256_20261101
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe compare.py --config configs/eme_long_memory_v2_full_rx_transfer.json --pretrained pretrained/full_rx_transfer_baseline256_20261101/model_best.pt --transfer-checkpoint pretrained/full_rx_transfer_post256_20261101/model_best.pt --snrs 0 5 10 15 --seed-start 60001 --num-seeds 5 --frames 8 --state-split heldout_edge --device cuda --output-dir logs/full_rx_transfer_four_arm256_5s8f_20261101
```
