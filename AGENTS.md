# AGENTS.md

我是中文语境开发者，所有对话和代码注释都用中文。

本仓库当前唯一研究路线是 Level B 极端稀疏长回波信道下的 Continual PPO 块均衡。任何新代码、测试和文档必须遵守以下契约：

- 主论文场景是 Level B；Level C 只做压力测试，不混入主平均。
- 接收机是整帧缓冲、非因果块均衡器；在线适配按帧进行。
- 全程 Pilot 条件监督预训练，配合 first-order meta 与 Best Fixed 前置门槛。
- Continual PPO 是第一主贡献，目标是进一步降低 `BER_data`，正式目标为每个主配置 `BER_data < 0.01`。
- Reward Pilot 只用于动作后的 reward 与留出评估；Data 标签只用于离线 outer loss 和仿真评估。
- 在线 observation、reward、动作选择、PEFT 更新不使用数据标签上界。
- 不恢复旧 A2C、旧逐符号 PPO、信道编码、多载波、多调制、MIMO、RIS、CFO 或非线性路线。
- 代码入口以 `pretrain.py`、`online_train.py`、`compare.py`、`calibrate_channel.py` 和 `pytest` 为准。

默认验证命令：

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/final_smoke
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/final_smoke/model_best.pt --frames 65 --num-seeds 1 --update-interval 32 --amp --output-dir logs/final_online_smoke
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --pretrained pretrained/final_smoke/model_best.pt --policy logs/final_online_smoke/policy.pt --delays 20 40 --snrs 10 20 --num-seeds 1 --frames 2 --output-dir logs/final_compare_smoke
```
