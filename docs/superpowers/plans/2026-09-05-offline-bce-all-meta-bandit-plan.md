# 整帧离线训练与在线适配调度实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans (inline execution). Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Level B 极端稀疏长回波信道上建立可审计的“整帧离线监督网络 + 可选元学习适配器 + 安全 Contextual Bandit 调度器”，并用配对实验判断元学习和延迟 RL 是否有必要。

**Architecture:** 离线训练对完整帧输出使用统一 `BCE_all`，但模型输入仍只接收 Adapt Pilot 可见符号。在线适配器只在 Adapt Pilot 上产生候选 PEFT 更新，Reward Pilot 负责独立验收和回滚。Bandit 只从有限安全动作中选择更新对象、强度和保持帧数；先通过延迟效应统计决定是否实现 Recurrent Double DQN。

**Tech Stack:** Python 3.12、PyTorch、NumPy、pytest、现有 `CommunicationEnvironment`、`UnfoldedEqualizer` 和 Level B EME 配置。

---

### Task 1: 固化离线 BCE_all 契约

**Files:** `training/curriculum.py`、`tests/test_meta_adaptation.py`、`docs/eme_offline_training_findings.md`

- [ ] **Step 1: Write the failing test.** 增加 `test_full_frame_bce_loss_weights_all_symbols_equally`，构造三个区域 BCE 不同的 logits，断言 `full_frame_bce_loss` 等于对完整张量直接调用 `binary_cross_entropy_with_logits`，且不依赖三个区域 mask。
- [ ] **Step 2: Run RED.** 运行 `\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_meta_adaptation.py::test_full_frame_bce_loss_weights_all_symbols_equally`，预期因函数不存在失败。
- [ ] **Step 3: Implement.** 在 `training/curriculum.py` 增加 `full_frame_bce_loss(logits, bits)`，对展平整帧使用 BCE；让 `_step_loss` 默认调用该函数，只有显式 `offline_loss="partitioned"` 才保留旧分区损失。传给模型的监督上下文使用 `frame.receiver_view().adapt_symbols`，使整帧标签只进入损失。
- [ ] **Step 4: Run GREEN.** 运行该单测和 `tests/test_meta_adaptation.py`。
- [ ] **Step 5: Commit.** `git add training/curriculum.py tests/test_meta_adaptation.py docs/eme_offline_training_findings.md; git commit -m "feat: 使用整帧 BCE 训练离线均衡器"`

### Task 2: 建立元学习必要性配对评估

**Files:** `training/online_meta_adaptation.py`、`training/online_adaptation.py`、`scripts/evaluate_meta_necessity.py`、`tests/test_online_meta_adaptation.py`

- [ ] **Step 1: Write the failing test.** 增加测试，使用同一帧和同一初始 checkpoint 比较普通 `PilotDrivenOnlineAdapter` 与元适配器，断言记录包含 Adapt/Reward loss、`data_labels_used_online=False`、步数和参数组，并能按 `early/middle/late` 聚合。
- [ ] **Step 2: Run RED.** 运行 `\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_online_meta_adaptation.py::test_meta_necessity_report_has_paired_frame_bins`，预期因报告 API 不存在失败。
- [ ] **Step 3: Implement.** 增加 `build_meta_necessity_report(rows)`，固定比较 Frozen、Pilot-SGD、Meta-Pilot，按配置、seed 和帧号配对，输出全程、前 16 帧、后 16 帧、heldout edge 的 Data BER 与配对差值。只有多 seed 后期 Data BER 稳定下降才标记元学习为 recommended。
- [ ] **Step 4: Run GREEN.** 运行元学习测试及 smoke，确认 Data 标签不进入在线状态、更新和动作选择。
- [ ] **Step 5: Commit.** `git add training/online_meta_adaptation.py training/online_adaptation.py scripts/evaluate_meta_necessity.py tests/test_online_meta_adaptation.py; git commit -m "feat: 增加元学习在线适配必要性评估"`

### Task 3: 收紧安全 Contextual Bandit 接口

**Files:** `agent/safe_contextual_bandit.py`、`training/online_adaptation.py`、`tests/test_safe_contextual_bandit.py`、`tests/test_pilot_online_adaptation.py`

- [ ] **Step 1: Write the failing test.** 断言默认上下文包含 residual CFO、phase slope、CIR drift、SNR、连续拒绝数和参数变化范数；低置信度只能返回 `skip` 或弱动作；动作 hold 窗口内保持不变；reward 更新不接收 Data 标签字段。
- [ ] **Step 2: Run RED.** 运行 `\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider tests/test_safe_contextual_bandit.py tests/test_pilot_online_adaptation.py`，预期新增审计断言失败。
- [ ] **Step 3: Implement.** 扩展稳定归一化上下文；runner 按动作窗口缓存动作，在 Reward Pilot 上计算 `loss_before - loss_after - update_cost - rollback_penalty`。Bandit 仍不直接修改参数；history 记录动作、上下文、reward、accepted、delta norm 和数据标签边界。
- [ ] **Step 4: Run GREEN.** 运行两个测试文件和全量 pytest。
- [ ] **Step 5: Commit.** `git add agent/safe_contextual_bandit.py training/online_adaptation.py tests/test_safe_contextual_bandit.py tests/test_pilot_online_adaptation.py; git commit -m "feat: 收紧安全上下文 bandit 的在线审计边界"`

### Task 4: 诊断动作是否具有跨帧延迟效应

**Files:** `scripts/diagnose_action_delay.py`、`tests/test_action_delay_diagnostic.py`、`README.md`、`docs/eme_action_delay_diagnostic.md`

- [ ] **Step 1: Write the failing test.** 给定按帧排列的动作和 Reward Pilot 统计，断言诊断器输出 horizon `0/1/2/4/8` 的平均 reward、配对增益和动作计数，并将 Data BER 标记为离线诊断列。
- [ ] **Step 2: Run RED.** 运行该测试，预期因诊断器不存在而失败。
- [ ] **Step 3: Implement.** 实现 `summarize_action_delay(rows, horizons=(0,1,2,4,8))`，从动作帧聚合未来 Reward Pilot reward，输出即时/延迟增益、bootstrap 置信区间和 `delayed_effect_detected`。延迟收益不显著高于即时收益时固定 Bandit；只有多 seed、多 SNR 稳定显著时才创建 DRQN 任务。
- [ ] **Step 4: Run GREEN.** 运行诊断单测和脚本 smoke，并把结论写入文档。
- [ ] **Step 5: Commit.** `git add scripts/diagnose_action_delay.py tests/test_action_delay_diagnostic.py README.md docs/eme_action_delay_diagnostic.md; git commit -m "feat: 增加在线动作跨帧延迟效应诊断"`

### Task 5: 正式复核、文档和推送

**Files:** `docs/eme_online_meta_adaptation_findings.md`、`docs/eme_guarded_online_state_recovery_results.md`、`docs/eme_three_layer_research_route.md`

- [ ] **Step 1: Run verification.** 运行 `\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider`、默认 pretrain smoke、默认 online smoke 和 compare smoke。
- [ ] **Step 2: Review and commit.** 运行 `git diff --check`、`git status --short`，只暂存本任务文件和文档，提交 `docs: 整理三层在线均衡研究路线`。
- [ ] **Step 3: Push.** 运行 `git push`，记录实际测试、实验目录、元学习是否必要、Bandit 是否足够和是否存在 delayed effect；不把未验证的 BER 优势写成结论。
