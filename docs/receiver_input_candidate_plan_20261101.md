# 接收输入与验收输入的四组合候选组件实验

此前离线审计确认，置零输入会改变 Pilot 代理，但没有证明实际候选收益。本轮
准备可审计的隔离组件，供确定正式接收信息边界前评估。它不是正式在线方法，
不修改默认主控制器、网络结构或 RL 动作，不将未答复的信息边界选择视为同意。
实际主协议仍待既有选择答复；当前研究契约保持不变。

## 看新输出前固定

- 源固定为 `pretrained/phase_transfer_post256_20261101/model_best.pt`，SHA256：
  `305ea6f636cf5f0619649489a7d2e5ed5ce9138334e37e731bf24246a78fe010`。
- 不重训基础权重，head/phase 两个已有组分别实验，候选不原地写回模型。
- 参数与此前相同：Adapt BCE、AdamW、lr=5e-4、1 步、增量上限 0.05。
- 每个组交叉四种协议：更新输入 Adapt-only / 整帧接收波形，验收输入
  Pilot-only / 整帧接收波形。相同更新候选复用给两种验收输入，分开两个因素。
- 输入波形范围与损失标签范围明确分离。两种更新输入始终只用 Adapt 已知符号
  构造损失，验收只用 Reward 已知符号，不把未知发送符号输入网络。
- 所有组合的候选和验收结束后，Data 标签才用于事后仿真指标；不使用 Data
  梯度驱动候选，不跨组合用 Data 选动作。
- offline_train 新 seed=42000–42002，每条 4 帧、四主 SNR，48 个上下文、
  2 组 × 4 协议，共 384 条记录。每帧重置，不含 PPO 或跨帧参数累计。
- acquisition、Adapt 条件与旧 tail 规则不变，逐帧波形及 Frozen 指标需匹配。
- 汇总各组合相对同 checkpoint Frozen 的 BER 改善，以及相对原严格输入组合
  的增量；以 seed 为单位计算配对区间，同时报告有害验收、修复/引入错误。

可微内循环与真实 AdamW 候选函数须对两种输入分别逐参数匹配。污染 Reward/Data
标签及未知发送符号不应改变候选；整帧接收输入允许对 Data 接收波形敏感。
默认路径仍需通过原隐藏波形污染不变测试。显式选择 full_rx 的信号重构调用
应拒绝，避免误将仅针对 BCE 实现的协议扩展到其他目标。

该 checkpoint 的历史可微训练仍采用严格输入，不能把本实验描述成已完成与
新输入匹配的离线元训练。若出现稳定候选收益，后续还需匹配输入的普通监督 /
更新后训练对照、独立确认、累计与正式 RL。若未出现，不扩大同配置留出矩阵。

```powershell
$env:RL4EQ_LINEAR_OPERATOR_BACKEND = 'fft'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& D:/Research/RL4EQ/.venv-gpu/Scripts/python.exe scripts/diagnose_receiver_input_updates.py --config configs/eme_long_memory_v2_phase_transfer.json --pretrained pretrained/phase_transfer_post256_20261101/model_best.pt --seed-start 42000 --num-seeds 3 --frames 4 --device cuda --output logs/receiver_input_candidates_3s4f_20261101.json
```
