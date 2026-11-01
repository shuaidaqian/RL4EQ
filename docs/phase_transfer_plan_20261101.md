# 现有神经相位分支的离线更新后泛化训练

## 假设与授权范围

三臂作用诊断表明：现有 `phase` 神经参数组比 head 更能改变 Data 连续输出，
但普通 Adapt BCE 尚未形成跨 seed 稳定 BER 收益。本轮检验外层训练能否改善
这个已有分支的 Pilot 到 Data 迁移，不新增结构、不注册正式 RL 动作。
这是在现有结构内的组件实验，不依赖新增结构的偏好选择；完整主方法仍需后续
动作注册、跨帧累计和 PPO 调度验证。

## 看新测试数据前固定的设置

- 源 checkpoint：上一阶段已固定的
  `pretrained/head_transfer_post256_20261101/model_best.pt`，SHA256 为
  `b8281a8132759865b1b9c303aed60067eaae72a6917f1a0ace6457c4f5b4b7e5`。
  A/B 从同一源开始，不能把此前 head 训练开销省略为从零训练预算。
- A 为继续 BCE_all；B 为 BCE_all + 更新后 Data BCE，权重均为 1。
  若 B 显示额外收益，再加不含内循环的同权重 Data 目标分支排除重加权解释。
- 内循环：现有 phase 组、Adapt-only BCE、AdamW、lr=5e-4、1 步、增量上限
  0.05；不使用后记忆评分变体，因为诊断没有支持其稳定优势。
- 两分支各 256 步、batch=1、外层 lr=1e-4，训练 seed=37000–37255，
  offline_train、四主 SNR 轮换；不改变信道、扰动、Pilot 或历史规则。
- FFT、float32、math attention。GPU 两步 smoke 与 phase 外层有限差分核验
  必须通过后才执行完整预算；不使用 AMP 或测试集选 checkpoint。
- 首次独立四臂测试：heldout_edge，seed=48001–48005、每条 8 帧、四主 SNR，
  每帧参数重置。验证单次迁移后才讨论累计和 PPO；短实验不代替长期主门槛。

## 判读

同时检查 B 相对自身 Frozen 的改善、B Online 相对 A Online 的绝对改善和
两分支适配收益差。B Frozen 退化造成更大的适配幅度不能作为成功。
Data 标签只用于离线外层和动作结束后的仿真评估，Reward 不进入新外层目标。
若单次迁移仍未成立，不扩大此配置矩阵，转向表示与可观测状态的重新设计。

## 复现

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe pretrain.py --config configs/eme_long_memory_v2_phase_transfer.json --stage pilot_transfer --transfer-objective baseline --resume pretrained/head_transfer_post256_20261101/model_best.pt --steps 256 --batch-size 1 --device cuda --save-dir pretrained/phase_transfer_baseline256_20261101
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe pretrain.py --config configs/eme_long_memory_v2_phase_transfer.json --stage pilot_transfer --transfer-objective post_adapt --resume pretrained/head_transfer_post256_20261101/model_best.pt --steps 256 --batch-size 1 --device cuda --save-dir pretrained/phase_transfer_post256_20261101
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe compare.py --config configs/eme_long_memory_v2_phase_transfer.json --pretrained pretrained/phase_transfer_baseline256_20261101/model_best.pt --transfer-checkpoint pretrained/phase_transfer_post256_20261101/model_best.pt --snrs 0 5 10 15 --seed-start 48001 --num-seeds 5 --frames 8 --state-split heldout_edge --device cuda --output-dir logs/phase_transfer_four_arm256_5s8f_20261101
```
