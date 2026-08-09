# AGENTS.md

我是中文语境开发者，所有对话和代码注释都用中文。

本仓库当前唯一研究路线是 Level B 极端稀疏长回波信道下的 `RL-Modulated Neural Block Equalizer`。任何新代码、测试和文档必须遵守以下契约：

- 主论文场景是 Level B；Level C 只做压力测试，不混入主平均。
- 主论文 SNR 固定为 0/5/10/15 dB。
- Pilot 只放在前缀；two-block/multi-block 只能作为历史兼容或诊断，不作为主实验。
- 接收机是整帧缓冲、非因果块神经均衡器；在线适配按帧进行。
- Proposed 方法是唯一使用神经网络和 RL 的方法。
- RL 选择离散安全动作，调制神经均衡器 Adapter/FiLM/LoRA/head；同一动作作用于短窗口，reward 使用窗口级 Reward Pilot BER/loss 改善；不逐 bit 判决，不直接输出完整高维参数增量。
- 传统 baseline 不使用神经网络，不使用 RL，只允许使用 acquisition/Adapt Pilot、接收信号和传统自适应规则。
- `Perfect-CSI Block` 与 `Fixed CG-BPSK-DD Block Detector` 只作为诊断参考，不作为主 baseline，不纳入主成功门槛。
- Reward Pilot 只用于动作后的 reward 与留出评估；Data 标签只用于离线监督和仿真评估。
- 在线 observation、reward、动作选择和调制更新不使用数据标签上界。
- 正式目标为 proposed 在每个 Level B 主配置达到 `BER_data < 0.01`，并超过传统非神经、非 RL baseline。
- 不恢复旧 A2C、旧逐符号 PPO、Data Oracle、多载波、MIMO 或 RIS 路线。
- CFO、额外相位扰动、非线性、信道编码和高阶调制只作为后续按需扩展开关；当前主实验默认关闭。
- 代码入口以 `pretrain.py`、`online_train.py`、`compare.py`、`calibrate_channel.py` 和 `pytest` 为准。

默认验证命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/final_smoke
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/final_smoke/model_best.pt --frames 8 --num-seeds 1 --window-size 4 --update-interval 4 --delays 20 --snrs 10 --pilot-total 64 --pilot-layout prefix --amp --output-dir logs/final_online_smoke
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --pretrained pretrained/final_smoke/model_best.pt --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 64 --pilot-layout prefix --resume --output-dir logs/final_compare_smoke
```
