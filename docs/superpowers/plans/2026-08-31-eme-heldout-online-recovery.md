# EME Held-Out 在线恢复实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在不改变 Level B EME 长记忆 profile、最大时延和 Pilot 协议的前提下，构造离线未见但物理合法的慢状态轨迹，验证 Pilot 在线元适配能否超过冻结离线模型。

**架构：** 将当前信道的隐状态范围作为显式 `state_split` 配置，由环境在 episode 初始化时确定性解析。离线训练、held-out 测试和 drift 测试共享同一个 profile 结构与物理上界，只改变当前 tap/CFO/相位实例及其轨迹。在线适配仍只使用 Adapt Pilot，Reward Pilot 只做 reward/guard，Data 标签只做离线监督和最终仿真统计。

**技术栈：** Python、PyTorch、pytest、现有 `CommunicationEnvironment`、现有 EME profile 和在线元适配入口。

---

### 任务 1：增加可审计的 EME 状态划分

**文件：**
- 创建：`env/online_state.py`
- 修改：`env/experiment_config.py`
- 修改：`env/comm_env.py`
- 修改：`env/extreme_delay_channel.py`
- 修改：`configs/eme_long_memory_v2.json`
- 测试：`tests/test_online_state_split.py`

- [ ] **步骤 1：先写失败测试**

测试 `offline_train`、`heldout_edge`、`drift` 三个状态划分具有确定性、边界合法、共享固定物理 profile，并拒绝覆盖 `max_delay_seconds`、Pilot layout 和 profile 名称。

- [ ] **步骤 2：运行测试确认失败**

运行：` .\.venv-gpu\Scripts\python.exe -m pytest -q tests/test_online_state_split.py -p no:cacheprovider`

预期：因状态划分解析器不存在而失败。

- [ ] **步骤 3：实现最小状态划分解析器**

使用不可变数据结构保存 `strong_path_count`、`diffuse_energy_ratio`、CFO 范围、相位噪声范围和 drift 参数；所有采样使用 episode seed 派生的 RNG，并将实际采样值写入环境元数据。

- [ ] **步骤 4：运行定向测试确认通过**

运行同一命令，预期测试全部通过，并确认每个 split 的 `max_delay_samples=116`、`profile_name=eme_long_memory_v2`、`pilot_layout=prefix`。

- [ ] **步骤 5：提交**

运行：`git add env/online_state.py env/experiment_config.py env/comm_env.py env/extreme_delay_channel.py configs/eme_long_memory_v2.json tests/test_online_state_split.py`，然后提交 `feat: 增加 EME 在线状态划分`。

### 任务 2：接入连续慢漂移轨迹

**文件：**
- 修改：`env/extreme_delay_channel.py`
- 修改：`env/comm_env.py`
- 修改：`tests/test_online_state_split.py`

- [ ] **步骤 1：先写失败测试**

测试同一 episode 内强径 support 不变，tap gain、CFO 和慢相位按配置递推；同 seed 可复现，不同 seed 产生不同轨迹；漂移轨迹不暴露真实状态给 `receiver_view`。

- [ ] **步骤 2：运行测试确认失败**

运行：` .\.venv-gpu\Scripts\python.exe -m pytest -q tests/test_online_state_split.py -p no:cacheprovider`

- [ ] **步骤 3：实现最小轨迹递推**

保留现有跨帧 history 和 soft tail 递推，新增只在 channel 内部维护的状态；快速相位/CFO 和慢 tap 采用不同更新系数。环境输出只包含接收帧、已知 Pilot 和已有 receiver state。

- [ ] **步骤 4：运行测试确认通过**

预期 drift 测试通过，且 `Frame.receiver_view()` 不包含 `true_cir`、真实 CFO 或真实相位。

- [ ] **步骤 5：提交**

提交 `feat: 接入 EME 连续慢漂移轨迹`。

### 任务 3：建立无 RL 的 held-out 在线比较

**文件：**
- 修改：`training/online_adaptation.py`
- 修改：`compare.py`
- 创建：`configs/eme_long_memory_v2_online_recovery.json`
- 测试：`tests/test_online_recovery_compare.py`

- [ ] **步骤 1：先写失败测试**

测试比较入口能够生成 paired 的 `Frozen Offline NN`、`Pilot-Driven Online Adaptation` 和两个传统 baseline，所有方法使用相同 seed、相同状态轨迹和相同前缀 Pilot。

- [ ] **步骤 2：运行测试确认失败**

运行：` .\.venv-gpu\Scripts\python.exe -m pytest -q tests/test_online_recovery_compare.py -p no:cacheprovider`

- [ ] **步骤 3：实现比较入口**

增加 `state_split`、`trajectory_id`、`drift_step`、`online_update_source`、`data_labels_used_online` 和每帧 Reward Pilot 指标。主比较不使用真实 CIR，诊断输出单独标记。

- [ ] **步骤 4：运行 smoke**

使用 1 个 seed、4 帧、10 dB 运行 held-out 和 drift 两个 smoke，确认 JSONL 中 `(method, seed, frame, trajectory_id)` 唯一。

- [ ] **步骤 5：提交**

提交 `feat: 增加 EME held-out 在线恢复比较`。

### 任务 4：增加 Safe Contextual Bandit 调度器

**文件：**
- 创建：`agent/safe_contextual_bandit.py`
- 修改：`training/online_adaptation.py`
- 修改：`compare.py`
- 测试：`tests/test_safe_contextual_bandit.py`

- [ ] **步骤 1：先写失败测试**

测试 Bandit 只接收 Pilot/历史 reward 特征，动作属于固定离散安全集合，安全盾牌拒绝超出更新强度或明显劣于 baseline 的动作。

- [ ] **步骤 2：运行测试确认失败**

运行：` .\.venv-gpu\Scripts\python.exe -m pytest -q tests/test_safe_contextual_bandit.py -p no:cacheprovider`

- [ ] **步骤 3：实现保守上下文 Bandit**

使用小型动作集合和每动作 ridge/UCB 统计，reward 为 Reward Pilot 前后 loss 改善减去更新代价；Bandit 只调度 Head/FiLM 更新强度和跨帧保持窗口，不直接输出网络参数。

- [ ] **步骤 4：运行定向测试和 smoke**

预期能在固定安全动作和 Bandit 动作之间完成可复现比较，且数据标签审计字段始终为 `false`。

- [ ] **步骤 5：提交**

提交 `feat: 增加安全上下文 bandit 在线调度`。

### 任务 5：正式验证和研究文档

**文件：**
- 创建：`docs/eme_online_recovery_results.md`
- 修改：`README.md`
- 修改：`AGENTS.md`（仅在契约需要补充时）

- [ ] **步骤 1：运行完整测试**

运行：` .\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider`

- [ ] **步骤 2：运行 held-out 小矩阵**

固定 Level B、SNR `0/5/10/15`、至少 5 个 seed 和连续帧，报告 Frozen、普通 Pilot、元适配、固定调度、Bandit 及传统 baseline。

- [ ] **步骤 3：检查结论门槛**

只有在 held-out/drift 下 Online 稳定超过 Frozen 且逐配置超过最强公平传统方法时，才记录“在线恢复成立”；否则诚实记录为“在线安全保持性能”，不把 Bandit 接受率包装成性能增益。

- [ ] **步骤 4：整理文档并提交**

记录数据划分、信息边界、逐配置结果、置信区间、失败案例和后续论文表述，提交 `docs: 记录 EME held-out 在线恢复结果`。

- [ ] **步骤 5：推送**

运行：`git push RL4EQ codex/eme-slow-drift-physical-rl`。
