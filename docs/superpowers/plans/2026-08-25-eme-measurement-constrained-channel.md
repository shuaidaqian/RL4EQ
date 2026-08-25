# EME Measurement-Constrained Channel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立由公开月球雷达文献约束、具有物理时间尺度、固定稀疏支撑、弱弥散长尾、慢复增益和连续跨帧 ISI 的 EME Level A/B/C 信道，并在神经网络训练前完成独立统计校准。

**Architecture:** 新增独立的 `eme_reference` 与 `eme_channel_profiles` 模块，不把文献数据塞入现有通用采样器。`ExtremeDelayChannel` 通过显式 `profile_name` 选择新 profile，并保留旧默认值用于历史测试。主 EME 配置从 `sample_rate_hz` 和 `max_delay_seconds` 计算离散时延，信道先连续处理符号流，再由环境切帧。

**Tech Stack:** Python 3.12、NumPy、PyTorch、pytest、JSON/CSV、Matplotlib（校准图）。

---

## 文件结构

- Create: `data/eme/evans_1965_fig8_envelope.csv`：Evans 1965 图 8 的可审计数字化锚点与上下包络。
- Create: `data/eme/reference_manifest.json`：文献、DOI、图号、载频/波长、数字化方法和用途。
- Create: `env/eme_reference.py`：参考数据加载、物理时延换算和包络插值。
- Create: `env/eme_channel_profiles.py`：稀疏强径、异常散射径和弱弥散尾的随机 profile。
- Modify: `env/extreme_delay_channel.py`：接入物理 EME profile 与按秒定义的慢状态。
- Modify: `env/comm_env.py`：向信道传递物理 profile 配置。
- Create: `configs/eme_measurement_channel_candidates.json`：仅含文献/方法约束的待校准候选。
- Create: `configs/continual_ppo_eme_measurement_v1.json`：冻结后正式实验配置。
- Create: `scripts/calibrate_eme_measurement_channel.py`：无 NN/RL 的信道统计和传统基线校准入口。
- Create: `tests/test_eme_reference_channel.py`：物理换算、数据溯源、稀疏度、慢状态和跨帧连续性测试。
- Modify: `tests/test_channel_protocol.py`：验证 receiver view 信息边界不因新 profile 改变。
- Create: `docs/eme_measurement_channel_calibration.md`：记录证据、校准结果、冻结参数与限制。

### Task 1: EME 参考数据与物理时间换算

**Files:**
- Create: `data/eme/evans_1965_fig8_envelope.csv`
- Create: `data/eme/reference_manifest.json`
- Create: `env/eme_reference.py`
- Test: `tests/test_eme_reference_channel.py`

- [ ] **Step 1: 写参考数据失败测试**

```python
from env.eme_reference import (
    EME_FULL_RADAR_DEPTH_SECONDS,
    load_evans_1965_envelope,
    physical_delay_samples,
)


def test_eme_reference_has_provenance_and_full_radar_depth():
    envelope = load_evans_1965_envelope()
    assert EME_FULL_RADAR_DEPTH_SECONDS == pytest.approx(0.0116)
    assert envelope.source_doi == "10.6028/jres.069d.195"
    assert envelope.delay_seconds[0] == 0.0
    assert envelope.delay_seconds[-1] == pytest.approx(0.0116)
    assert np.all(np.diff(envelope.delay_seconds) > 0.0)
    assert np.all(envelope.lower_power_db <= envelope.upper_power_db)


def test_physical_delay_is_derived_from_sample_rate():
    assert physical_delay_samples(2_000.0, 0.0116) == 24
    assert physical_delay_samples(10_000.0, 0.0116) == 116
```

- [ ] **Step 2: 运行失败测试**

Run: `./.venv-gpu/Scripts/python.exe -m pytest tests/test_eme_reference_channel.py -q -p no:cacheprovider`

Expected: FAIL，提示 `env.eme_reference` 不存在。

- [ ] **Step 3: 写入参考数据和 manifest**

CSV 使用以下列：

```csv
delay_ms,power_db_3p6cm,power_db_68cm,source_figure,digitization_note
0.0,0.0,0.0,Evans1965-Fig8,normalized leading edge
...
11.6,-24.0,-40.0,Evans1965-Fig8,limb endpoint
```

manifest 必须记录 DOI、公开 PDF URL、图号、图轴、归一化方式、数字化日期和“只作为包络约束，不宣称 1.296 GHz 精确 PDP”。

- [ ] **Step 4: 实现只读参考加载器**

```python
@dataclass(frozen=True)
class EMEEchoEnvelope:
    delay_seconds: np.ndarray
    upper_power_db: np.ndarray
    lower_power_db: np.ndarray
    source_doi: str


def physical_delay_samples(sample_rate_hz: float, max_delay_seconds: float) -> int:
    if sample_rate_hz <= 0.0 or max_delay_seconds <= 0.0:
        raise ValueError("采样率和最大时延必须为正。")
    return int(math.ceil(sample_rate_hz * max_delay_seconds))
```

包络加载必须使用 `csv`/`json` 标准解析，不允许依赖工作目录。

- [ ] **Step 5: 运行参考数据测试**

Run: `./.venv-gpu/Scripts/python.exe -m pytest tests/test_eme_reference_channel.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 6: 提交参考数据阶段**

```powershell
git add data/eme env/eme_reference.py tests/test_eme_reference_channel.py
git commit -m "feat: 加入EME雷达时延参考数据"
```

### Task 2: 文献约束的稀疏/弥散 EME profile

**Files:**
- Create: `env/eme_channel_profiles.py`
- Modify: `tests/test_eme_reference_channel.py`

- [ ] **Step 1: 写 profile 失败测试**

```python
from env.eme_channel_profiles import EMEChannelProfileConfig, sample_eme_profile


def test_level_b_eme_profile_is_sparse_long_and_reproducible():
    config = EMEChannelProfileConfig(
        level="B",
        sample_rate_hz=2_000.0,
        symbol_rate_hz=2_000.0,
        frame_len=512,
        strong_path_count=(3, 7),
        diffuse_energy_ratio=(0.05, 0.15),
        seed=17,
    )
    first = sample_eme_profile(config)
    second = sample_eme_profile(config)
    assert first.max_delay_samples == 24
    assert first.strong_delays == second.strong_delays
    assert np.allclose(first.cir, second.cir)
    assert first.strong_delays[0] == 0
    assert max(first.strong_delays) >= 6
    assert 3 <= len(first.strong_delays) <= 7
    assert 0.05 <= first.diffuse_energy_ratio <= 0.15
    assert np.isclose(np.sum(np.abs(first.cir) ** 2), 1.0)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./.venv-gpu/Scripts/python.exe -m pytest tests/test_eme_reference_channel.py::test_level_b_eme_profile_is_sparse_long_and_reproducible -q -p no:cacheprovider`

Expected: FAIL，提示模块不存在。

- [ ] **Step 3: 实现配置和结果类型**

```python
@dataclass(frozen=True)
class EMEChannelProfileConfig:
    level: str
    sample_rate_hz: float
    symbol_rate_hz: float
    frame_len: int
    strong_path_count: tuple[int, int]
    diffuse_energy_ratio: tuple[float, float]
    seed: int
    max_delay_seconds: float = EME_FULL_RADAR_DEPTH_SECONDS


@dataclass(frozen=True)
class EMEChannelProfile:
    cir: np.ndarray
    strong_delays: tuple[int, ...]
    diffuse_mask: np.ndarray
    diffuse_energy_ratio: float
    effective_taps_90: int
    max_delay_samples: int
    metadata: dict[str, object]
```

- [ ] **Step 4: 实现采样规则**

采样规则必须满足：

- delay 0 是前沿强径；
- 其余强径按 EME 包络加权采样，至少一条位于最大支撑的后 25%；
- 允许一个异常散射径相对当地包络提升，提升范围以 Pettengill/Henry 的 7--8 倍局部均值观测为依据；
- Level B 弥散尾能量由配置显式给定，不在模块内隐藏硬编码；
- 弥散尾相位随机、平均功率服从参考包络；
- 最后统一归一化，并计算 `effective_taps_90`。

- [ ] **Step 5: 加入 Level A/B/C 分类测试**

验证 A 无或极弱弥散尾，B 为主，C 只返回 `aggregation="pressure"` 且弥散能量更高。

- [ ] **Step 6: 运行 profile 测试**

Run: `./.venv-gpu/Scripts/python.exe -m pytest tests/test_eme_reference_channel.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 7: 提交 profile 阶段**

```powershell
git add env/eme_channel_profiles.py tests/test_eme_reference_channel.py
git commit -m "feat: 实现文献约束的EME稀疏长回波profile"
```

### Task 3: 接入连续跨帧信道和物理慢状态

**Files:**
- Modify: `env/extreme_delay_channel.py`
- Modify: `env/comm_env.py`
- Modify: `tests/test_eme_reference_channel.py`
- Modify: `tests/test_channel_protocol.py`

- [ ] **Step 1: 写物理配置与跨帧脉冲失败测试**

```python
def test_eme_channel_derives_delay_and_carries_impulse_into_next_frame():
    channel = ExtremeDelayChannel(
        ExtremeDelayChannelConfig(
            profile_name="eme_measurement_v1",
            level="B",
            sample_rate_hz=2_000.0,
            symbol_rate_hz=2_000.0,
            frame_len=16,
            coherence_time_seconds=120.0,
            snr_db=80.0,
            seed=31,
        )
    )
    assert channel.max_delay == 24
    channel.reset_episode(torch.zeros(24, dtype=torch.complex64))
    first = torch.zeros(16, dtype=torch.complex64)
    first[-1] = 1.0 + 0.0j
    channel.transmit(first, add_noise=False)
    second = channel.transmit(torch.zeros(32, dtype=torch.complex64), add_noise=False)
    expected_delays = [delay - 1 for delay in channel.delays if delay > 0]
    assert all(second[index].abs() > 0.0 for index in expected_delays)
```

- [ ] **Step 2: 写 frame 相关系数失败测试**

```python
def test_rho_frame_is_derived_from_frame_duration_and_coherence_time():
    cfg = ExtremeDelayChannelConfig(
        profile_name="eme_measurement_v1",
        sample_rate_hz=2_000.0,
        symbol_rate_hz=2_000.0,
        frame_len=512,
        coherence_time_seconds=120.0,
    )
    assert cfg.frame_duration_seconds == pytest.approx(0.256)
    assert cfg.rho_frame == pytest.approx(math.exp(-0.256 / 120.0))
```

- [ ] **Step 3: 运行测试确认失败**

Run: `./.venv-gpu/Scripts/python.exe -m pytest tests/test_eme_reference_channel.py -q -p no:cacheprovider`

Expected: FAIL，提示新配置字段不存在。

- [ ] **Step 4: 最小接入新 profile**

`ExtremeDelayChannelConfig` 新增：

```python
profile_name: str = "legacy_sparse_v1"
sample_rate_hz: float | None = None
symbol_rate_hz: float | None = None
frame_len: int = 512
max_delay_seconds: float = EME_FULL_RADAR_DEPTH_SECONDS
coherence_time_seconds: float | None = None
diffuse_energy_ratio: tuple[float, float] | None = None
```

仅当 `profile_name == "eme_measurement_v1"` 时要求物理字段完整，并计算 `max_delay`、`frame_duration_seconds` 和 `rho_frame`。当前发送链路每个复样本就是一个符号，必须验证 `sample_rate_hz == symbol_rate_hz`，并显式暴露 `samples_per_symbol == 1.0`；不得在没有脉冲成形和匹配滤波的情况下伪装 2 samples/symbol。旧 profile 保留当前行为，避免无关回归。

- [ ] **Step 5: 用完整 CIR 执行连续卷积**

`_convolve_with_history` 必须遍历非零完整 CIR，而不是只遍历强径列表，以包含弥散尾；历史长度使用计算出的 `max_delay`。

- [ ] **Step 6: 更新环境配置透传**

`CommEnvConfig` 增加同名字段，`CommunicationEnvironment` 只负责透传，不重新计算信道参数。

- [ ] **Step 7: 验证信息边界**

在 `tests/test_channel_protocol.py` 中使用新 EME profile 重跑 receiver view 测试，确认真实 CIR、CFO、相位状态和 Data 标签均不可见。

- [ ] **Step 8: 运行信道相关测试**

Run: `./.venv-gpu/Scripts/python.exe -m pytest tests/test_eme_reference_channel.py tests/test_channel_protocol.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 9: 提交信道接入阶段**

```powershell
git add env/extreme_delay_channel.py env/comm_env.py tests/test_eme_reference_channel.py tests/test_channel_protocol.py
git commit -m "feat: 接入物理时标和跨帧EME长回波"
```

### Task 4: 候选配置与无神经统计校准

**Files:**
- Create: `configs/eme_measurement_channel_candidates.json`
- Create: `scripts/calibrate_eme_measurement_channel.py`
- Modify: `tests/test_eme_reference_channel.py`

- [ ] **Step 1: 写候选配置 schema 测试**

验证候选只扫描 `strong_path_count`、`diffuse_energy_ratio`、`coherence_time_seconds` 和 residual impairment 预算；固定 `max_delay_seconds=0.0116`、SNR `0/5/10/15`、prefix Pilot，并标记 `traditional_only=true`。

- [ ] **Step 2: 写校准输出失败测试**

```python
def test_calibration_smoke_reports_channel_statistics_without_proposed(tmp_path):
    payload = run_calibration(candidates=[candidate], seeds=[0, 1], frames=4, output_dir=tmp_path)
    assert payload["traditional_only"] is True
    assert payload["proposed_methods_included"] is False
    assert payload["max_delay_seconds"] == pytest.approx(0.0116)
    assert payload["statistics"][0]["support_changed_frames"] == 0
    assert "effective_taps_90" in payload["statistics"][0]
    assert "frame_lag_correlation" in payload["statistics"][0]
    assert "envelope_rmse_db" in payload["statistics"][0]
```

- [ ] **Step 3: 实现统计校准入口**

校准输出至少包含：

- 物理与离散最大时延；
- 强径数量和支撑；
- 90% 能量有效抽头数；
- 弥散能量比；
- 经验 frame-lag 相关系数；
- 与参考包络的 RMSE；
- 跨帧脉冲响应误差；
- 传统 baseline 按 SNR 的 BER；
- 是否使用 NN/RL 的显式布尔字段。

- [ ] **Step 4: 运行 smoke 校准测试**

Run: `./.venv-gpu/Scripts/python.exe -m pytest tests/test_eme_reference_channel.py -q -p no:cacheprovider`

Expected: PASS。

- [ ] **Step 5: 提交校准入口**

```powershell
git add configs/eme_measurement_channel_candidates.json scripts/calibrate_eme_measurement_channel.py tests/test_eme_reference_channel.py
git commit -m "feat: 加入EME信道统计校准入口"
```

### Task 5: 冻结 profile、文档和全量回归

**Files:**
- Create: `configs/continual_ppo_eme_measurement_v1.json`
- Create: `docs/eme_measurement_channel_calibration.md`
- Modify: `tests/test_eme_reference_channel.py`

- [ ] **Step 1: 运行中等规模传统-only 校准**

Run:

```powershell
./.venv-gpu/Scripts/python.exe scripts/calibrate_eme_measurement_channel.py `
  --config configs/eme_measurement_channel_candidates.json `
  --seeds 0 1 2 3 4 `
  --frames 100 `
  --output-dir logs/eme_measurement_channel_calibration_2026-08-25
```

Expected: 生成 JSON、CSV 和参考包络/经验 PDP 图；输出中不含 Proposed 方法。

- [ ] **Step 2: 按预先声明的统计门槛冻结候选**

选择顺序固定为：物理时延正确、支撑稳定、跨帧误差为零、慢相关满足配置、参考包络误差最小、最后才检查传统 BER 是否处于可学习但不饱和区。不得读取任何 NN/RL 结果。

- [ ] **Step 3: 写冻结配置测试**

验证正式配置包含：

```json
{
  "channel_profile": "eme_measurement_v1",
  "max_delay_seconds": 0.0116,
  "sample_rate_hz": 2000.0,
  "symbol_rate_hz": 2000.0,
  "main_snrs": [0, 5, 10, 15],
  "pilot_layout": "prefix",
  "profile_frozen_before_proposed": true
}
```

- [ ] **Step 4: 写校准文档**

文档记录文献证据、数字化曲线、候选范围、所有失败候选、冻结理由、已知限制和“稀疏统计来自类 EME 方法而非月面直接测量”的声明。

- [ ] **Step 5: 运行信道测试和全量回归**

Run:

```powershell
./.venv-gpu/Scripts/python.exe -m pytest tests/test_eme_reference_channel.py tests/test_channel_protocol.py tests/test_traditional_baselines.py -q -p no:cacheprovider
./.venv-gpu/Scripts/python.exe -m pytest -q -p no:cacheprovider
```

Expected: 全部 PASS。

- [ ] **Step 6: 提交冻结 profile 阶段**

```powershell
git add configs/continual_ppo_eme_measurement_v1.json docs/eme_measurement_channel_calibration.md tests/test_eme_reference_channel.py
git commit -m "docs: 冻结EME测量约束信道profile"
```

## 完成条件

- 所有物理时延都可从秒和采样率追溯；
- Level B 支撑跨帧固定，复增益慢变；
- 跨帧 ISI 由连续卷积精确产生；
- 平均 PDP 受 Evans/Winter 公开数据约束；
- 稀疏/弥散统计来源和 EME 物理证据严格分开；
- profile 在运行任何 Proposed 训练前冻结；
- 全量 pytest 通过；
- 下一阶段才能编写 Offline NN 复现计划。
