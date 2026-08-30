# EME 在线元适配 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 EME 在线均衡主线从普通 Pilot PEFT 微调升级为显式优化 Adapt Pilot 更新后 Reward Pilot 性能的序列化一阶元适配，并保持现有信息边界和入口兼容。

**Architecture:** 新增独立的元适配训练器，复用现有 `PilotDrivenOnlineAdapter` 的受限更新逻辑，但用可微的内循环更新产生 post-adaptation logits，再以 Reward Pilot 和跨帧 soft-tail 状态计算外层损失。离线入口通过配置选择元训练阶段；在线入口继续执行一到两步安全 PEFT 更新，后续调度器不进入本阶段主实现。

**Tech Stack:** Python、PyTorch、pytest、现有 `UnfoldedEqualizer`、`CommunicationEnvironment`、`Frame` 和 `model_config` 机制。

---

### Task 1: 固化设计和训练契约

**Files:**
- Create: `docs/superpowers/specs/2026-08-31-eme-online-meta-adaptation-design.md`
- Create: `docs/superpowers/plans/2026-08-31-eme-online-meta-adaptation.md`
- Modify: `docs/eme_long_memory_v2_warmstart_findings.md`

- [ ] **Step 1: 记录研究假设和可证伪门槛**

明确区分匹配分布 sanity check、有限状态失配主实验和慢漂移压力测试；固定 Level B、prefix Pilot 和四个主 SNR。

- [ ] **Step 2: 检查设计文档中的信息边界**

确认在线只使用 Adapt Pilot、Reward Pilot、接收信号和 profile-level 先验，Data 标签只用于离线训练和仿真评估。

- [ ] **Step 3: 提交设计文档**

运行 `git add docs/superpowers/specs docs/superpowers/plans`，提交信息使用 `docs: 固化EME在线元适配设计`。

### Task 2: 为序列化元训练编写失败测试

**Files:**
- Create: `tests/test_online_meta_adaptation.py`
- Inspect: `training/meta_training.py`
- Inspect: `training/online_adaptation.py`

- [ ] **Step 1: 测试 Adapt Pilot 更新不读取 Data 标签**

构造同一接收帧、不同 Data 标签但相同 Pilot 的输入，断言 inner update 的参数更新和 pre/post Adapt Pilot loss 不变，并检查审计字段为 `False`。

- [ ] **Step 2: 测试 post-adaptation Reward Pilot 外层损失真实计算**

使用一个可控小模型和人工 Frame，断言元训练结果同时返回 `pre_adapt_reward_loss`、`post_adapt_reward_loss` 和 `meta_loss`，且 post loss 来自留出 Reward Pilot。

- [ ] **Step 3: 测试跨帧 soft tail 顺序不被更新回滚破坏**

让第二帧的输入依赖第一帧输出，触发第一帧参数拒绝后，断言接收机保留最新 tail，而不是恢复到窗口起点。

- [ ] **Step 4: 运行定向测试确认按预期失败**

运行 `\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_online_meta_adaptation.py`，预期因元适配 API 尚不存在而失败。

### Task 3: 实现可测试的内循环和外层 Reward Pilot 目标

**Files:**
- Create or Modify: `training/online_meta_adaptation.py`
- Modify: `training/online_adaptation.py`
- Test: `tests/test_online_meta_adaptation.py`

- [ ] **Step 1: 增加 PEFT 快照和恢复的可组合接口**

复用 `PEFTRegistry` 的 group 解析；快照只包含选定 PEFT 参数，恢复不触碰 soft tail 和已确认的物理状态。

- [ ] **Step 2: 实现无 optimizer state 的一阶 inner update**

对 Adapt Pilot BCE 计算梯度，按学习率和最大增量范数更新 selected PEFT 参数；支持 `create_graph=False` 的一阶模式，避免把二阶图带入 smoke 训练。

- [ ] **Step 3: 实现 post-adaptation Reward Pilot loss**

更新前计算 Reward Pilot loss，执行 inner update 后重新前向，返回 pre/post loss、参数增量和 meta loss；缺少 Reward Pilot 时显式报错而不是静默改用 Data。

- [ ] **Step 4: 实现序列状态推进**

按 frame index 顺序运行当前帧，接受更新后推进 soft tail，拒绝更新时只恢复 PEFT；将每帧审计字段写入结构化结果。

- [ ] **Step 5: 运行定向测试确认通过**

运行 `\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_online_meta_adaptation.py`，预期全部通过。

### Task 4: 接入离线入口和 EME 配置

**Files:**
- Modify: `pretrain.py`
- Modify: `training/curriculum.py` or `training/meta_training.py`
- Modify: `env/experiment_config.py`
- Modify: `configs/eme_long_memory_v2.json`
- Modify: `tests/test_experiment_config.py`

- [ ] **Step 1: 增加元训练配置开关**

增加 `online_meta_training`、`meta_inner_steps`、`meta_inner_lr`、`meta_outer_weight` 和 `meta_sequence_frames`，默认关闭以保持旧配置兼容，EME 配置显式开启。

- [ ] **Step 2: 将训练样本划分为 Adapt/Reward/Data**

训练阶段使用 Adapt Pilot 计算内循环，Reward Pilot 计算外层目标，Data 只计算离线辅助损失；prefix layout 不改变。

- [ ] **Step 3: 增加配置维度和标签边界测试**

断言元训练不接受非 prefix 主配置，且在线结果字段中 `data_labels_used_online` 永远为 `False`。

- [ ] **Step 4: 运行入口 smoke**

运行 `\.venv-gpu\Scripts\python.exe pretrain.py --config configs/eme_long_memory_v2.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/eme_meta_smoke`，确认生成 checkpoint 和 `model_config.json`。

### Task 5: 在线算法对照实验和统计

**Files:**
- Modify: `compare.py`
- Modify: `evaluation/research_diagnostics.py`
- Create: `docs/eme_online_meta_adaptation_findings.md`
- Test: `tests/test_evaluation_contract.py`

- [ ] **Step 1: 接入 Frozen、普通 PEFT 和元适配三种神经方法标识**

确保三者使用相同信道、相同 prefix Pilot、相同物理 warm-start；只有元适配使用 post-adaptation 训练出的 checkpoint。

- [ ] **Step 2: 增加逐帧统计字段**

记录 Adapt/Reward loss、接受率、参数增量、CFO/phase 置信度、tail 更新率、前五帧和后五帧 BER。

- [ ] **Step 3: 运行单配置短矩阵**

运行 `\.venv-gpu\Scripts\python.exe compare.py --config configs/eme_long_memory_v2.json --method-group main --pretrained pretrained/eme_meta_smoke/model_best.pt --delays 116 --snrs 10 --num-seeds 2 --frames 8 --pilot-total 128 --pilot-layout prefix --resume --output-dir logs/eme_meta_short`。

- [ ] **Step 4: 写入阶段文档**

如元适配只改善 Reward Pilot 但不改善 Data，必须如实记录；不得使用 Data BER 反向选择更新策略或信道参数。

### Task 6: 全量验证、提交和推送

**Files:**
- Modify: `docs/eme_long_memory_v2_warmstart_findings.md`
- Modify: `docs/eme_online_meta_adaptation_findings.md`

- [ ] **Step 1: 运行完整测试**

运行 `\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider` 并记录通过数和失败数。

- [ ] **Step 2: 检查差异和缓存文件**

只提交源代码、测试、配置和文档；不要加入 `agent/__pycache__`、`env/__pycache__` 等遗留缓存变化。

- [ ] **Step 3: 提交实现和阶段结果**

使用中文提交信息，例如 `feat: 加入EME序列化在线元适配`。

- [ ] **Step 4: 推送当前分支**

运行 `git push RL4EQ codex/eme-slow-drift-physical-rl`，然后核对远程分支 HEAD 与本地提交一致。
