# EME 冻结 Profile 端到端透传实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让离线预训练、在线窗口 PPO 和正式对比实验实际运行冻结的 `eme_measurement_v1/B-core` 信道，并对任何会改变冻结信道的 CLI/配置冲突立即报错。

**Architecture:** 新增 `env.experiment_config` 作为实验 JSON 到 `CommEnvConfig` 的唯一适配边界。legacy 配置保持原行为；EME 配置严格校验所有物理字段、由物理时标计算并核对 `D`、强制 prefix Pilot，并把实际生效的信道元数据写入训练/在线/对比产物。

**Tech Stack:** Python 3.12、PyTorch、pytest、JSON/JSONL

---

### Task 1: 冻结配置适配器

**Files:**
- Create: `env/experiment_config.py`
- Create: `tests/test_experiment_config.py`

- [x] **Step 1: 写入失败测试**

测试 `build_comm_env_config()` 对 `eme_measurement_v1` 逐字段透传，并断言 `profile_name/sample_rate_hz/symbol_rate_hz/frame_len/max_delay_seconds/coherence_time_seconds/strong_path_count/diffuse_energy_ratio/include_anomalous_scatterer` 与冻结配置一致。

- [x] **Step 2: 运行测试并确认因模块不存在而失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_experiment_config.py -q -p no:cacheprovider`

Expected: FAIL，错误为 `ModuleNotFoundError: env.experiment_config`。

- [x] **Step 3: 实现最小适配器**

公开以下接口：

```python
def build_comm_env_config(
    experiment: Mapping[str, Any], *, level: str, snr_db: float, seed: int,
    max_delay: int | None = None, total_pilot: int | None = None,
    pilot_layout: str | None = None, impairment_profile: str | None = None,
) -> CommEnvConfig: ...

def effective_channel_metadata(config: CommEnvConfig) -> dict[str, Any]: ...
```

EME 模式缺字段、`D` 冲突、非 prefix、模型维度冲突或 `eme_physical_fields_passthrough != "implemented"` 时抛出中文 `ValueError`；legacy 模式保持旧默认值。

- [x] **Step 4: 运行适配器测试并确认通过**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_experiment_config.py -q -p no:cacheprovider`

Expected: PASS。

### Task 2: 离线预训练入口透传

**Files:**
- Modify: `training/curriculum.py`
- Modify: `tests/test_continual_ppo.py`

- [x] **Step 1: 写入失败测试**

用冻结配置构造 `CurriculumTrainer`，拦截其训练样本、Level B 校验和离线 NN 校验环境，断言三条路径均为 `eme_measurement_v1`；同时断言 EME curriculum 不再生成 Level A/旧 delay 网格。

- [x] **Step 2: 运行测试并确认旧代码产生 legacy 环境而失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_continual_ppo.py -q -p no:cacheprovider`

- [x] **Step 3: 用统一适配器替换三处手写配置**

EME 配置只构造冻结 Level B 的 SNR 网格；legacy curriculum 保持历史四阶段行为。预训练 metrics 增加 `effective_channel`。

- [x] **Step 4: 运行测试并确认通过**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_continual_ppo.py tests/test_experiment_config.py -q -p no:cacheprovider`

### Task 3: 在线 PPO 与对比入口透传

**Files:**
- Modify: `training/windowed_discrete_ppo.py`
- Modify: `compare.py`
- Modify: `tests/test_windowed_discrete_ppo.py`
- Modify: `tests/test_evaluation_contract.py`

- [x] **Step 1: 写入失败测试**

分别运行 1 帧 EME smoke，断言在线 `online_metrics.json`、对比 `summary.json/frame_metrics.jsonl` 记录实际 `eme_measurement_v1` 及物理字段；传入 `--delays 40` 或 `--pilot-layout two_block` 时必须失败。

- [x] **Step 2: 运行测试并确认旧入口静默使用 legacy 而失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_windowed_discrete_ppo.py tests/test_evaluation_contract.py -q -p no:cacheprovider`

- [x] **Step 3: 替换环境构造并写入实际元数据**

在线与 compare 均调用 `build_comm_env_config()`；输出中的 `delay/profile_name/effective_channel` 来自已构造环境配置，不从文件名或未校验 CLI 回填。

- [x] **Step 4: 运行测试并确认通过**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_windowed_discrete_ppo.py tests/test_evaluation_contract.py tests/test_experiment_config.py -q -p no:cacheprovider`

### Task 4: 冻结契约、阶段文档和回归

**Files:**
- Modify: `configs/continual_ppo_eme_measurement_v1.json`
- Modify: `docs/eme_measurement_channel_calibration.md`
- Modify: `tests/test_eme_reference_channel.py`

- [x] **Step 1: 将 `eme_physical_fields_passthrough` 改为 `implemented` 并更新测试**

配置必须继续引用版本化冻结证据及其 SHA-256；文档明确三个正式入口已经端到端使用 B-core。

- [x] **Step 2: 运行 EME 契约与三个 smoke**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_eme_reference_channel.py tests/test_experiment_config.py -q -p no:cacheprovider`

Run: `.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo_eme_measurement_v1.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/eme_passthrough_smoke`

Run: `.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo_eme_measurement_v1.json --pretrained pretrained/eme_passthrough_smoke/model_best.pt --frames 1 --num-seeds 1 --window-size 1 --update-interval 1 --delays 24 --snrs 10 --pilot-total 128 --pilot-layout prefix --amp --output-dir logs/eme_passthrough_online_smoke`

Run: `.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo_eme_measurement_v1.json --method-group traditional --delays 24 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 128 --pilot-layout prefix --output-dir logs/eme_passthrough_compare_smoke`

- [x] **Step 3: 运行全量回归**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider`

- [x] **Step 4: 分阶段提交**

只暂存适配器、三个入口、对应测试、冻结配置和阶段文档；不触碰工作树中其他既有修改。
