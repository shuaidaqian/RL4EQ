# RL4EQ：极端长时延块均衡中的 Continual PPO 研究路线

本项目研究 EME 启发的极端稀疏长回波信道下的单载波 BPSK 整帧块均衡。核心目标不是在常规无线信道里替代成熟传统均衡器，而是在传统方法明显受限的 20–40 符号长时延扩展场景中，验证物理展开神经接收机与部署期间持续更新 PPO 的增益边界。

当前唯一主线：

```text
Level A/B/C 可控信道族
-> 全程 Pilot 条件课程监督预训练
-> First-order meta 初始化与 Best Fixed 前置门槛
-> 整帧缓冲、非因果 Physics-Guided Unfolded Equalizer
-> Continual PPO 在信道运行期间按帧持续更新策略
-> 与传统、固定微调、规则控制和 Bandit baseline 配对比较
```

研究契约：

- 主论文场景是 Level B：20/30/40 符号最大相对时延，SNR=10/15/20 dB，稀疏强长回波，`rho=0.99`。
- Level A 用于课程学习和可达性校准；Level C 只作为压力测试，不混入 Level B 主平均，也不要求所有 Level C 配置达到 `BER_data < 0.01`。
- 接收机是整帧缓冲、非因果块神经均衡器；“在线”指信道运行期间按帧持续适配，不是逐符号即时判决式输出。
- 全程使用 Pilot 条件监督预训练；Pilot 开销/布局按 `64/96/128/160 × prefix/two_block/multi_block` 做消融。
- Reward Pilot 只用于动作完成后的 reward 与留出评估；Data 标签只用于离线监督 outer loss 和仿真 `BER_data` 评估。
- 在线 observation、reward、动作选择和参数更新不使用数据标签上界。
- Continual PPO 的第一目标是进一步降低 `BER_data`；前置门槛是每个主配置 Best Fixed 低于 0.1，PPO 目标是每个主配置 `BER_data < 0.01`。

## 当前代码入口

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/final_smoke
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage meta --steps 2 --batch-size 1 --resume pretrained/final_smoke/last.pt --save-dir pretrained/meta_smoke
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/meta_smoke/model_best.pt --frames 65 --num-seeds 1 --update-interval 32 --amp --output-dir logs/final_online_smoke
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --pretrained pretrained/meta_smoke/model_best.pt --policy logs/final_online_smoke/policy.pt --delays 20 40 --snrs 10 20 --num-seeds 1 --frames 2 --output-dir logs/final_compare_smoke
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

GPU 环境安装：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup_gpu_env.ps1 -Recreate
.\.venv-gpu\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
```

PyTorch wheel 自带 CUDA runtime；脚本不修改系统 CUDA Toolkit。

## 目录结构

```text
agent/
  cir_estimator.py              Hybrid Sparse CIR Estimator
  unfolded_equalizer.py         Physics-Guided Unfolded Equalizer
  continual_policy.py           分层 recurrent mixed-action PPO policy
  adaptation_controller.py      PEFT 动作执行、shadow reward、安全回滚
baseline/
  block_equalizers.py           Perfect/Pilot-estimated 块均衡与迭代检测
  channel_tracking.py           Sparse-LS、Kalman/RLS 跟踪接口
  legacy_equalizers.py          传统 LMMSE-FIR 与 DFE 标签隔离 baseline
env/
  channel_profiles.py           Level A/B/C 信道族
  extreme_delay_channel.py      长回波、漂移、跨帧 ISI
  frame_structure.py            多 Pilot 子块帧结构
  comm_env.py                   acquisition + 普通帧 episode 环境
training/
  curriculum.py                 Pilot 条件课程监督预训练
  meta_training.py              first-order episodic meta 与 Best Fixed gate
  continual_ppo.py              部署期间持续更新 PPO runner
evaluation/
  metrics.py                    BER、goodput、Spearman、逐配置汇总
  bootstrap.py                  seed + 连续帧块 bootstrap
  runner.py                     分片与 resume 支撑
```

## 阶段门槛

```text
Perfect-CSI 强检测器：Level B 九配置分别 BER_data < 0.01
Perfect-CIR Unfolded：Level B 九配置分别 BER_data < 0.01
Best Fixed Adaptation：Level B 九配置分别 BER_data < 0.1
Reward/Data Spearman：>= 0.6
Continual PPO：Level B 九配置分别 BER_data < 0.01
```

任何前置门槛失败时，必须停在对应阶段定位假设，不能用总体平均或 PPO 后处理掩盖接收机不可达。

## 当前实现边界

当前分支已建立可运行的 GPU smoke、协议测试和评估契约。部分训练/比较入口仍是面向 smoke 与工程契约验证的轻量实现；正式论文结论必须来自后续完整开发规模与正式 10 seed × 1000 frame 配对矩阵。未真实重跑前，不写“显著优于传统算法”或“PPO 显著优于固定微调”。
