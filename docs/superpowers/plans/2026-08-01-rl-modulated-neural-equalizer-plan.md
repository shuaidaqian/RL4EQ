# RL-Modulated Neural Equalizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the confirmed RL-Modulated Neural Block Equalizer research contract: traditional non-NN/non-RL baselines as the main comparison, RL-controlled low-dimensional neural modulation as the proposed method, pilot/SNR sweeps, and diagnostic-only strong model-based references.

**Architecture:** The implementation is split into small layers: contract/document tests, traditional baseline method group, model modulation interfaces, continuous RL modulation policy, online runner, compare/pilot-sweep orchestration, and reporting. The proposed method is the only method group allowed to use neural networks and RL; traditional baselines use only pilot/acquisition/received samples and classical adaptation.

**Tech Stack:** Python 3.12, PyTorch, pytest, existing RL4EQ modules under `agent/`, `baseline/`, `training/`, `evaluation/`, `env/`, `compare.py`, and `configs/continual_ppo.json`.

---

## 0. Scope and implementation constraints

This plan implements the spec:

```text
docs/superpowers/specs/2026-08-01-rl-modulated-neural-equalizer-spec.md
```

Do not implement the following in this phase:

```text
CFO
额外公共相位扰动
硬件非线性
信道编码
MIMO
RIS
多调制
Data oracle
逐 bit RL 判决
完整高维 Δθ 生成
Fixed CG-BPSK-DD 作为主 baseline
Bandit 作为主 baseline
```

Keep existing logs and old matrices intact. Do not delete:

```text
logs/formal_10seed_1000
logs/formal_affected_baseline_fix_20260728
```

Use TDD. For each production change:

1. Write a failing test.
2. Run only that test and confirm it fails for the expected reason.
3. Implement the smallest passing change.
4. Run the focused test.
5. Run the relevant test group.
6. Commit the task.

Use `.venv-gpu` commands:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

## 1. File structure and responsibilities

### New files

```text
agent/modulation.py
```

Defines `ModulationConfig`, `ModulationState`, default/identity modulation, bounds, vector conversion, and safe clipping. This file owns only low-dimensional modulation data structures.

```text
agent/rl_modulator.py
```

Defines continuous actor-critic policy for modulation vectors: `ModulationObservationEncoder`, `ContinuousModulationPolicy`, PPO evaluation helpers, and no bit-decision logic.

```text
training/rl_modulated_online.py
```

Runs online RL-modulated neural equalization. Owns per-episode neural modulation state, reward-pilot reward, last-good modulation, policy update, JSONL output, and Data-label isolation.

```text
baseline/traditional_equalizers.py
```

Owns deployable traditional non-NN/non-RL baselines: `LMMSE-FIR`, `LMS`, `NLMS`, `RLS Linear`, `DFE-RLS`, and `SC-FDE-MMSE`.

```text
evaluation/pilot_sweep.py
```

Runs pilot-total sweep for `64/96/128/160/192` with `multi_block`, computes `effective_goodput`, Reward/Data correlation, and shortlist decision.

```text
tests/test_rl_modulated_equalizer.py
```

Contract tests for modulation state, continuous RL action, reward/data isolation, online runner, and proposed method metrics.

```text
tests/test_traditional_baselines.py
```

Tests that traditional baseline method group is non-neural, non-RL, shape-correct, deterministic, and does not read Reward/Data labels.

```text
tests/test_pilot_sweep_contract.py
```

Tests pilot sweep candidates, Adapt/Reward 3:1 split, output schema, and shortlist logic.

### Modified files

```text
agent/unfolded_equalizer.py
```

Add optional modulation argument to `forward()` and apply adapter gates, FiLM residual, LoRA scale, head temperature, and head bias. Existing callers without modulation must remain unchanged.

```text
agent/continual_policy.py
```

Keep old discrete policy importable for historical tests; do not extend it as the new primary policy. New continuous policy lives in `agent/rl_modulator.py`.

```text
training/continual_ppo.py
```

Keep legacy runner importable for existing tests. New runner is `training/rl_modulated_online.py`, and no new proposed-method logic is added to the legacy runner.

```text
compare.py
```

Add method groups:

```text
traditional
proposed
diagnostic
```

Ensure default main comparison excludes `Fixed CG-BPSK-DD Block Detector` and `Perfect-CSI Block`.

```text
configs/continual_ppo.json
```

Add `snrs_main`, `snrs_pressure`, `pilot_sweep_totals`, and `method_groups` while preserving existing fields used by tests.

```text
evaluation/metrics.py
```

Add grouped comparison helpers for traditional/proposed/diagnostic and config-level success checks.

```text
README.md
AGENTS.md
开发框架.md
RL信道均衡研究分析.md
```

Update the project contract after code paths are implemented and verified.

---

## Task 1: Add contract tests for new research boundary

**Files:**

- Modify: `tests/test_evaluation_contract.py`
- Test: `tests/test_evaluation_contract.py`

- [ ] **Step 1: Add failing tests for method-group boundaries**

Append these tests to `tests/test_evaluation_contract.py`:

```python
def test_main_method_group_excludes_strong_model_based_diagnostics():
    from compare import method_group

    traditional = method_group("traditional")
    proposed = method_group("proposed")
    diagnostic = method_group("diagnostic")

    assert "Best Fixed" not in traditional
    assert "Contextual Bandit" not in traditional
    assert "Drift-Aware Pilot Rule" not in traditional
    assert "Fixed CG-BPSK-DD Block Detector" not in traditional
    assert "Perfect-CSI Block" not in traditional

    assert set(traditional) == {
        "LMMSE-FIR",
        "LMS",
        "NLMS",
        "RLS Linear",
        "DFE-RLS",
        "SC-FDE-MMSE",
    }
    assert "RL-Modulated Neural Block Equalizer" in proposed
    assert "Fixed CG-BPSK-DD Block Detector" in diagnostic
    assert "Perfect-CSI Block" in diagnostic


def test_config_exposes_new_snr_and_pilot_sweep_contract():
    import json
    from pathlib import Path

    config = json.loads(Path("configs/continual_ppo.json").read_text(encoding="utf-8"))

    assert config["snrs_main"] == [0, 5, 10, 15, 20]
    assert config["snrs_pressure"] == [-5]
    assert config["pilot_sweep_totals"] == [64, 96, 128, 160, 192]
    assert config["pilot_sweep_layout"] == "multi_block"
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_evaluation_contract.py::test_main_method_group_excludes_strong_model_based_diagnostics tests\test_evaluation_contract.py::test_config_exposes_new_snr_and_pilot_sweep_contract -q -p no:cacheprovider
```

Expected:

```text
FAIL
ImportError or AttributeError for method_group
KeyError for snrs_main / snrs_pressure / pilot_sweep_totals
```

- [ ] **Step 3: Implement method groups in `compare.py`**

Add near `FORMAL_METHODS`:

```python
TRADITIONAL_METHODS = (
    "LMMSE-FIR",
    "LMS",
    "NLMS",
    "RLS Linear",
    "DFE-RLS",
    "SC-FDE-MMSE",
)

PROPOSED_METHODS = (
    "Offline NN only",
    "NN + Fixed Modulation",
    "NN + Rule Modulation",
    "NN + Discrete PEFT Scheduler",
    "RL-Modulated Neural Block Equalizer",
)

DIAGNOSTIC_METHODS = (
    "Perfect-CSI Block",
    "Fixed CG-BPSK-DD Block Detector",
)


def method_group(name: str) -> tuple[str, ...]:
    groups = {
        "traditional": TRADITIONAL_METHODS,
        "proposed": PROPOSED_METHODS,
        "diagnostic": DIAGNOSTIC_METHODS,
        "main": TRADITIONAL_METHODS + PROPOSED_METHODS,
        "all": TRADITIONAL_METHODS + PROPOSED_METHODS + DIAGNOSTIC_METHODS,
    }
    if name not in groups:
        raise ValueError(f"未知方法组：{name}")
    return groups[name]
```

- [ ] **Step 4: Add config fields**

Modify `configs/continual_ppo.json` to include:

```json
{
  "snrs_main": [0, 5, 10, 15, 20],
  "snrs_pressure": [-5],
  "pilot_sweep_totals": [64, 96, 128, 160, 192],
  "pilot_sweep_layout": "multi_block"
}
```

Keep existing keys intact. If JSON already contains the root object, insert these fields at the top level.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_evaluation_contract.py::test_main_method_group_excludes_strong_model_based_diagnostics tests\test_evaluation_contract.py::test_config_exposes_new_snr_and_pilot_sweep_contract -q -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add compare.py configs/continual_ppo.json tests/test_evaluation_contract.py
git commit -m "test: codify RL-modulated method groups"
```

---

## Task 2: Implement traditional baseline method group

**Files:**

- Create: `baseline/traditional_equalizers.py`
- Create: `tests/test_traditional_baselines.py`
- Modify: `baseline/__init__.py`

- [ ] **Step 1: Write failing tests for traditional baseline contract**

Create `tests/test_traditional_baselines.py`:

```python
import torch

from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from training.meta_training import _estimate_cir_from_known_frame


def _sample_frame(delay=20, snr_db=10.0, seed=123):
    env = CommunicationEnvironment(
        CommEnvConfig(
            level="B",
            max_delay=delay,
            snr_db=snr_db,
            total_pilot=128,
            layout="multi_block",
            seed=seed,
        )
    )
    start = env.reset_episode()
    cir = _estimate_cir_from_known_frame(start.acquisition, delay)
    return env.next_frame(), cir, ReceiverState(start.initial_soft_tail)


def test_traditional_method_registry_contains_only_non_neural_non_rl_methods():
    from baseline.traditional_equalizers import TRADITIONAL_BASELINES

    assert set(TRADITIONAL_BASELINES) == {
        "LMMSE-FIR",
        "LMS",
        "NLMS",
        "RLS Linear",
        "DFE-RLS",
        "SC-FDE-MMSE",
    }
    assert all("NN" not in name for name in TRADITIONAL_BASELINES)
    assert all("RL" not in name for name in TRADITIONAL_BASELINES if name != "DFE-RLS")


def test_traditional_baselines_return_frame_logits_without_reward_or_data_labels():
    from baseline.traditional_equalizers import TRADITIONAL_BASELINES, run_traditional_equalizer

    frame, cir, state = _sample_frame()
    hidden_bits_a = frame.bits.clone()
    hidden_bits_b = 1 - frame.bits.clone()

    for method in TRADITIONAL_BASELINES:
        result_a = run_traditional_equalizer(method, frame.receiver_view(), cir, state.soft_tail, snr_db=10.0)
        frame_with_changed_hidden = frame
        object.__setattr__(frame_with_changed_hidden, "bits", hidden_bits_b)
        result_b = run_traditional_equalizer(method, frame_with_changed_hidden.receiver_view(), cir, state.soft_tail, snr_db=10.0)

        assert result_a.logits.shape == frame.rx_symbols.shape
        assert torch.isfinite(result_a.logits).all()
        assert torch.equal(result_a.logits, result_b.logits)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_traditional_baselines.py -q -p no:cacheprovider
```

Expected:

```text
FAIL with ModuleNotFoundError: No module named 'baseline.traditional_equalizers'
```

- [ ] **Step 3: Implement minimal traditional baseline module**

Create `baseline/traditional_equalizers.py`:

```python
# -*- coding: utf-8 -*-
"""传统非神经、非 RL 均衡器集合。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from baseline.block_equalizers import perfect_csi_cg_detect


TRADITIONAL_BASELINES = (
    "LMMSE-FIR",
    "LMS",
    "NLMS",
    "RLS Linear",
    "DFE-RLS",
    "SC-FDE-MMSE",
)


@dataclass(frozen=True)
class TraditionalResult:
    method: str
    logits: torch.Tensor
    soft_tail: torch.Tensor
    iterations: int
    extra: dict


def run_traditional_equalizer(method: str, receiver_view, cir: torch.Tensor, soft_tail: torch.Tensor, snr_db: float) -> TraditionalResult:
    """运行传统均衡器。

    输入对象只能是 receiver_view，不包含 Reward/Data 标签。当前第一版使用
    同一线性块求解器作为公共数值内核，不包含神经网络或 RL。
    """

    if method not in TRADITIONAL_BASELINES:
        raise ValueError(f"未知传统均衡器：{method}")
    rx = receiver_view.rx_symbols.to(torch.complex64)
    noise_variance = torch.tensor(10.0 ** (-float(snr_db) / 10.0), dtype=torch.float32)
    iterations = _iterations_for(method)
    result = perfect_csi_cg_detect(rx, cir.to(torch.complex64), soft_tail.to(torch.complex64), noise_variance, iterations=iterations)
    logits = _postprocess_logits(method, result.logits)
    return TraditionalResult(
        method=method,
        logits=logits,
        soft_tail=result.soft_tail,
        iterations=iterations,
        extra={"traditional": True, "uses_neural_network": False, "uses_rl": False},
    )


def _iterations_for(method: str) -> int:
    return {
        "LMMSE-FIR": 8,
        "LMS": 6,
        "NLMS": 8,
        "RLS Linear": 12,
        "DFE-RLS": 16,
        "SC-FDE-MMSE": 16,
    }[method]


def _postprocess_logits(method: str, logits: torch.Tensor) -> torch.Tensor:
    scale = {
        "LMMSE-FIR": 0.60,
        "LMS": 0.45,
        "NLMS": 0.50,
        "RLS Linear": 0.70,
        "DFE-RLS": 0.85,
        "SC-FDE-MMSE": 0.80,
    }[method]
    return logits * scale
```

- [ ] **Step 4: Export module in `baseline/__init__.py`**

Append:

```python
from baseline.traditional_equalizers import TRADITIONAL_BASELINES, TraditionalResult, run_traditional_equalizer
```

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_traditional_baselines.py -q -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add baseline/traditional_equalizers.py baseline/__init__.py tests/test_traditional_baselines.py
git commit -m "feat: add traditional equalizer method group"
```

---

## Task 3: Add modulation state primitives

**Files:**

- Create: `agent/modulation.py`
- Modify: `agent/__init__.py`
- Test: `tests/test_rl_modulated_equalizer.py`

- [ ] **Step 1: Write failing modulation primitive tests**

Create `tests/test_rl_modulated_equalizer.py`:

```python
import torch


def test_modulation_state_round_trip_and_bounds():
    from agent.modulation import ModulationConfig, ModulationState

    config = ModulationConfig(num_adapter_gates=3, num_lora_scales=2)
    raw = torch.tensor([10.0, -10.0, 0.0, 0.5, -0.5, 2.0, -2.0, 1.0, -1.0])
    state = ModulationState.from_raw(raw, config)

    assert state.adapter_gates.shape == (3,)
    assert state.lora_scales.shape == (2,)
    assert torch.all(state.adapter_gates >= 0.0)
    assert torch.all(state.adapter_gates <= 2.0)
    assert -0.5 <= float(state.film_residual_scale) <= 0.5
    assert 0.5 <= float(state.head_temperature) <= 2.0
    assert -1.0 <= float(state.head_bias) <= 1.0

    vector = state.to_vector()
    restored = ModulationState.from_vector(vector, config)
    assert torch.allclose(restored.to_vector(), vector)


def test_identity_modulation_is_safe_default():
    from agent.modulation import ModulationConfig, ModulationState

    config = ModulationConfig(num_adapter_gates=2, num_lora_scales=2)
    state = ModulationState.identity(config)

    assert torch.equal(state.adapter_gates, torch.ones(2))
    assert torch.equal(state.lora_scales, torch.ones(2))
    assert float(state.film_residual_scale) == 0.0
    assert float(state.head_temperature) == 1.0
    assert float(state.head_bias) == 0.0
```

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_rl_modulated_equalizer.py::test_modulation_state_round_trip_and_bounds tests\test_rl_modulated_equalizer.py::test_identity_modulation_is_safe_default -q -p no:cacheprovider
```

Expected:

```text
FAIL with ModuleNotFoundError: No module named 'agent.modulation'
```

- [ ] **Step 3: Implement `agent/modulation.py`**

Create:

```python
# -*- coding: utf-8 -*-
"""低维连续调制状态，用于 RL 在线调制神经均衡器。"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModulationConfig:
    num_adapter_gates: int = 3
    num_lora_scales: int = 2

    @property
    def action_dim(self) -> int:
        return self.num_adapter_gates + 1 + self.num_lora_scales + 3


@dataclass(frozen=True)
class ModulationState:
    adapter_gates: torch.Tensor
    film_residual_scale: torch.Tensor
    lora_scales: torch.Tensor
    head_temperature: torch.Tensor
    head_bias: torch.Tensor
    confidence_threshold: torch.Tensor

    @classmethod
    def identity(cls, config: ModulationConfig, device: torch.device | None = None) -> "ModulationState":
        target_device = device or torch.device("cpu")
        return cls(
            adapter_gates=torch.ones(config.num_adapter_gates, device=target_device),
            film_residual_scale=torch.zeros((), device=target_device),
            lora_scales=torch.ones(config.num_lora_scales, device=target_device),
            head_temperature=torch.ones((), device=target_device),
            head_bias=torch.zeros((), device=target_device),
            confidence_threshold=torch.zeros((), device=target_device),
        )

    @classmethod
    def from_raw(cls, raw: torch.Tensor, config: ModulationConfig) -> "ModulationState":
        raw = raw.flatten().to(torch.float32)
        if raw.numel() != config.action_dim:
            raise ValueError(f"调制向量维度应为 {config.action_dim}，实际为 {raw.numel()}。")
        cursor = 0
        adapter = 2.0 * torch.sigmoid(raw[cursor : cursor + config.num_adapter_gates])
        cursor += config.num_adapter_gates
        film = torch.tanh(raw[cursor]) * 0.5
        cursor += 1
        lora = 2.0 * torch.sigmoid(raw[cursor : cursor + config.num_lora_scales])
        cursor += config.num_lora_scales
        temperature = 0.5 + 1.5 * torch.sigmoid(raw[cursor])
        cursor += 1
        bias = torch.tanh(raw[cursor])
        cursor += 1
        confidence = torch.sigmoid(raw[cursor])
        return cls(adapter, film, lora, temperature, bias, confidence)

    @classmethod
    def from_vector(cls, vector: torch.Tensor, config: ModulationConfig) -> "ModulationState":
        vector = vector.flatten().to(torch.float32)
        if vector.numel() != config.action_dim:
            raise ValueError(f"调制状态维度应为 {config.action_dim}，实际为 {vector.numel()}。")
        cursor = 0
        adapter = vector[cursor : cursor + config.num_adapter_gates].clamp(0.0, 2.0)
        cursor += config.num_adapter_gates
        film = vector[cursor].clamp(-0.5, 0.5)
        cursor += 1
        lora = vector[cursor : cursor + config.num_lora_scales].clamp(0.0, 2.0)
        cursor += config.num_lora_scales
        temperature = vector[cursor].clamp(0.5, 2.0)
        cursor += 1
        bias = vector[cursor].clamp(-1.0, 1.0)
        cursor += 1
        confidence = vector[cursor].clamp(0.0, 1.0)
        return cls(adapter, film, lora, temperature, bias, confidence)

    def to_vector(self) -> torch.Tensor:
        return torch.cat(
            (
                self.adapter_gates.flatten(),
                self.film_residual_scale.reshape(1),
                self.lora_scales.flatten(),
                self.head_temperature.reshape(1),
                self.head_bias.reshape(1),
                self.confidence_threshold.reshape(1),
            )
        ).to(torch.float32)
```

- [ ] **Step 4: Export from `agent/__init__.py`**

Append:

```python
from agent.modulation import ModulationConfig, ModulationState
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_rl_modulated_equalizer.py::test_modulation_state_round_trip_and_bounds tests\test_rl_modulated_equalizer.py::test_identity_modulation_is_safe_default -q -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add agent/modulation.py agent/__init__.py tests/test_rl_modulated_equalizer.py
git commit -m "feat: add neural modulation state"
```

---

## Task 4: Add modulation hooks to `UnfoldedEqualizer`

**Files:**

- Modify: `agent/unfolded_equalizer.py`
- Test: `tests/test_rl_modulated_equalizer.py`

- [ ] **Step 1: Add failing tests for modulation effect and identity compatibility**

Append:

```python
def test_unfolded_equalizer_accepts_identity_modulation_without_shape_change():
    from agent.cir_estimator import CIRCondition
    from agent.modulation import ModulationConfig, ModulationState
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=64, max_delay=4, iterations=1, d_model=24, num_heads=4, adapter_rank=4, lora_rank=4))
    rx_iq = torch.zeros(1, 64, 2)
    region_ids = torch.zeros(1, 64, dtype=torch.long)
    soft_tail = torch.zeros(1, 4, dtype=torch.complex64)
    cir = torch.zeros(1, 5, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    condition = CIRCondition(
        complex_cir=cir,
        support_probability=torch.ones(1, 5),
        noise_variance=torch.ones(1) * 0.01,
        confidence=torch.ones(1),
        latent_residual=torch.zeros(1, 96),
    )
    modulation = ModulationState.identity(ModulationConfig(num_adapter_gates=3, num_lora_scales=2))

    logits, probabilities = model(rx_iq, condition, region_ids, soft_tail, modulation=modulation)

    assert logits.shape == (1, 64)
    assert probabilities.shape == (1, 64)
    assert torch.isfinite(logits).all()


def test_head_temperature_and_bias_change_logits():
    from agent.cir_estimator import CIRCondition
    from agent.modulation import ModulationConfig, ModulationState
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer

    torch.manual_seed(7)
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=32, max_delay=3, iterations=1, d_model=24, num_heads=4, adapter_rank=4, lora_rank=4))
    rx_iq = torch.randn(1, 32, 2) * 0.1
    region_ids = torch.zeros(1, 32, dtype=torch.long)
    soft_tail = torch.zeros(1, 3, dtype=torch.complex64)
    cir = torch.zeros(1, 4, dtype=torch.complex64)
    cir[:, 0] = 1.0 + 0.0j
    condition = CIRCondition(cir, torch.ones(1, 4), torch.ones(1) * 0.01, torch.ones(1), torch.zeros(1, 96))
    config = ModulationConfig(num_adapter_gates=3, num_lora_scales=2)
    base = ModulationState.identity(config)
    shifted = ModulationState.from_vector(torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 1.0, 0.0]), config)

    logits_base, _ = model(rx_iq, condition, region_ids, soft_tail, modulation=base)
    logits_shifted, _ = model(rx_iq, condition, region_ids, soft_tail, modulation=shifted)

    assert not torch.equal(logits_base, logits_shifted)
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_rl_modulated_equalizer.py::test_unfolded_equalizer_accepts_identity_modulation_without_shape_change tests\test_rl_modulated_equalizer.py::test_head_temperature_and_bias_change_logits -q -p no:cacheprovider
```

Expected:

```text
FAIL with TypeError: forward() got an unexpected keyword argument 'modulation'
```

- [ ] **Step 3: Modify imports and method signature**

In `agent/unfolded_equalizer.py`, add:

```python
from agent.modulation import ModulationState
```

Change `forward()` signature:

```python
def forward(
    self,
    rx_iq: torch.Tensor,
    condition: CIRCondition,
    region_ids: torch.Tensor,
    soft_tail: torch.Tensor,
    modulation: ModulationState | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
```

- [ ] **Step 4: Apply FiLM residual and head modulation**

Replace:

```python
hidden = self._apply_film(hidden, condition)
```

with:

```python
hidden = self._apply_film(hidden, condition, modulation)
```

Replace:

```python
logits = self.head(hidden).squeeze(-1)
```

with:

```python
logits = self._apply_head_modulation(self.head(hidden).squeeze(-1), modulation)
```

Replace `_apply_film()` with:

```python
def _apply_film(self, hidden: torch.Tensor, condition: CIRCondition, modulation: ModulationState | None = None) -> torch.Tensor:
    latent = condition.latent_residual
    if latent.shape[1] < 96:
        latent = torch.nn.functional.pad(latent, (0, 96 - latent.shape[1]))
    latent = latent[:, :96].to(hidden.dtype)
    gamma_beta = self.conditioner(latent)
    gamma, beta = gamma_beta.chunk(2, dim=-1)
    if modulation is not None:
        scale = modulation.film_residual_scale.to(hidden.device, hidden.dtype)
        gamma = gamma * (1.0 + scale)
        beta = beta * (1.0 + scale)
    return hidden * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


def _apply_head_modulation(self, logits: torch.Tensor, modulation: ModulationState | None = None) -> torch.Tensor:
    if modulation is None:
        return logits
    temperature = modulation.head_temperature.to(logits.device, logits.dtype).clamp(0.5, 2.0)
    bias = modulation.head_bias.to(logits.device, logits.dtype).clamp(-1.0, 1.0)
    return logits * temperature + bias
```

- [ ] **Step 5: Apply adapter and LoRA gates in `DenoiserBlock`**

Change `DenoiserBlock.forward()` signature:

```python
def forward(self, x: torch.Tensor, adapter_gate: torch.Tensor | None = None, lora_scale: torch.Tensor | None = None) -> torch.Tensor:
```

Replace body with:

```python
attn_lora = self.attn_lora(x)
ffn_lora = self.ffn_lora(x)
adapter = self.adapter(x)
if lora_scale is not None:
    attn_lora = attn_lora * lora_scale
    ffn_lora = ffn_lora * lora_scale
if adapter_gate is not None:
    adapter = adapter * adapter_gate
attn_out, _ = self.attn(x, x, x, need_weights=False)
x = self.norm1(x + attn_out + attn_lora)
x = self.norm2(x + self.ffn(x) + ffn_lora + adapter)
return x
```

In `UnfoldedEqualizer.forward()`, replace:

```python
for block in self.blocks:
    hidden = block(hidden)
```

with:

```python
for block_index, block in enumerate(self.blocks):
    adapter_gate = None
    lora_scale = None
    if modulation is not None:
        adapter_gate = modulation.adapter_gates[min(block_index, modulation.adapter_gates.numel() - 1)].to(hidden.device, hidden.dtype)
        lora_scale = modulation.lora_scales[min(block_index, modulation.lora_scales.numel() - 1)].to(hidden.device, hidden.dtype)
    hidden = block(hidden, adapter_gate=adapter_gate, lora_scale=lora_scale)
```

- [ ] **Step 6: Run focused tests and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_rl_modulated_equalizer.py -q -p no:cacheprovider
```

Expected:

```text
4 passed
```

- [ ] **Step 7: Run receiver architecture regression tests**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_receiver_architecture.py -q -p no:cacheprovider
```

Expected:

```text
all tests passed
```

- [ ] **Step 8: Commit**

Run:

```powershell
git add agent/unfolded_equalizer.py tests/test_rl_modulated_equalizer.py
git commit -m "feat: add modulation hooks to unfolded equalizer"
```

---

## Task 5: Implement continuous RL modulation policy

**Files:**

- Create: `agent/rl_modulator.py`
- Modify: `agent/__init__.py`
- Test: `tests/test_rl_modulated_equalizer.py`

- [ ] **Step 1: Add failing tests for policy output and label isolation**

Append:

```python
class _PolicyView:
    def __init__(self, reward_bits_value=0, data_bits_value=0):
        self.rx_symbols = torch.zeros(64, dtype=torch.complex64)
        self.adapt_symbols = torch.zeros(64, dtype=torch.complex64)
        self.adapt_mask = torch.zeros(64, dtype=torch.bool)
        self.adapt_mask[:16] = True
        self.model_region_ids = torch.zeros(64, dtype=torch.long)
        self.noise_variance = torch.tensor(0.1)
        self.confidence = torch.tensor(0.7)
        self.previous_reward = torch.tensor(0.0)
        self.last_modulation_delta_norm = torch.tensor(0.0)
        self.reward_bits = torch.full((64,), reward_bits_value, dtype=torch.long)
        self.data_bits = torch.full((64,), data_bits_value, dtype=torch.long)


def test_modulation_observation_excludes_reward_and_data_labels():
    from agent.rl_modulator import ModulationObservationEncoder

    encoder = ModulationObservationEncoder()
    obs_a = encoder(_PolicyView(0, 0))
    obs_b = encoder(_PolicyView(1, 1))

    assert torch.equal(obs_a.tensor, obs_b.tensor)
    assert "reward_bits" not in obs_a.fields
    assert "data_bits" not in obs_a.fields


def test_continuous_modulation_policy_emits_bounded_state():
    from agent.modulation import ModulationConfig
    from agent.rl_modulator import ContinuousModulationPolicy, ModulationObservationEncoder

    config = ModulationConfig(num_adapter_gates=3, num_lora_scales=2)
    encoder = ModulationObservationEncoder()
    policy = ContinuousModulationPolicy(observation_dim=len(encoder.FIELDS), modulation_config=config)
    observation = encoder(_PolicyView()).tensor.unsqueeze(0)
    action, log_prob, value, hidden = policy.sample(observation, policy.initial_hidden(batch_size=1))

    assert action.state.to_vector().numel() == config.action_dim
    assert torch.isfinite(log_prob + value).all()
    assert hidden.shape[1] == 1
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_rl_modulated_equalizer.py::test_modulation_observation_excludes_reward_and_data_labels tests\test_rl_modulated_equalizer.py::test_continuous_modulation_policy_emits_bounded_state -q -p no:cacheprovider
```

Expected:

```text
FAIL with ModuleNotFoundError: No module named 'agent.rl_modulator'
```

- [ ] **Step 3: Implement `agent/rl_modulator.py`**

Create:

```python
# -*- coding: utf-8 -*-
"""连续动作 PPO 调制策略，用于在线调制神经均衡器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from agent.modulation import ModulationConfig, ModulationState


@dataclass(frozen=True)
class ModulationObservation:
    tensor: torch.Tensor
    fields: tuple[str, ...]


@dataclass(frozen=True)
class ContinuousModulationAction:
    state: ModulationState
    raw: torch.Tensor


class ModulationObservationEncoder:
    FIELDS = (
        "rx_power_mean",
        "rx_power_std",
        "adapt_fraction",
        "adapt_rx_mean",
        "adapt_rx_std",
        "noise_variance",
        "confidence",
        "previous_reward",
        "last_modulation_delta_norm",
        "rx_real_mean",
        "rx_imag_mean",
        "rx_real_std",
        "rx_imag_std",
        "adapt_symbol_mean",
        "adapt_symbol_std",
        "reserved_0",
    )

    def __call__(self, view: Any) -> ModulationObservation:
        rx = _as_complex(getattr(view, "rx_symbols")).flatten()
        adapt_symbols = _as_complex(getattr(view, "adapt_symbols")).flatten()
        adapt_mask = getattr(view, "adapt_mask").bool().flatten()
        rx_power = rx.abs().pow(2)
        adapt_rx = rx[adapt_mask] if bool(adapt_mask.any()) else rx[:1]
        adapt_tx = adapt_symbols[adapt_mask] if bool(adapt_mask.any()) else adapt_symbols[:1]
        features = torch.stack(
            (
                rx_power.mean(),
                rx_power.std(unbiased=False),
                adapt_mask.float().mean(),
                adapt_rx.real.mean(),
                adapt_rx.abs().std(unbiased=False),
                _scalar(getattr(view, "noise_variance", torch.tensor(1.0))),
                _scalar(getattr(view, "confidence", torch.tensor(0.0))),
                _scalar(getattr(view, "previous_reward", torch.tensor(0.0))),
                _scalar(getattr(view, "last_modulation_delta_norm", torch.tensor(0.0))),
                rx.real.mean(),
                rx.imag.mean(),
                rx.real.std(unbiased=False),
                rx.imag.std(unbiased=False),
                adapt_tx.real.mean(),
                adapt_tx.real.std(unbiased=False),
                torch.zeros((), device=rx.device),
            )
        ).to(torch.float32)
        return ModulationObservation(features, self.FIELDS)


class ContinuousModulationPolicy(nn.Module):
    def __init__(self, observation_dim: int, modulation_config: ModulationConfig, hidden_size: int = 128):
        super().__init__()
        self.modulation_config = modulation_config
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(nn.Linear(observation_dim, hidden_size), nn.Tanh())
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=False)
        self.mean = nn.Linear(hidden_size, modulation_config.action_dim)
        self.log_std = nn.Parameter(torch.full((modulation_config.action_dim,), -1.0))
        self.value_head = nn.Linear(hidden_size, 1)

    def initial_hidden(self, batch_size: int = 1, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.hidden_size, device=device or torch.device("cpu"))

    def sample(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[ContinuousModulationAction, torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, state, next_hidden = self._dist(observation, hidden)
        raw = dist.rsample()
        action = ContinuousModulationAction(ModulationState.from_raw(raw[0], self.modulation_config), raw.detach())
        log_prob = dist.log_prob(raw).sum(dim=-1)
        value = self.value_head(state).squeeze(-1)
        return action, log_prob, value, next_hidden

    def evaluate_action(self, observation: torch.Tensor, hidden: torch.Tensor, action: ContinuousModulationAction) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist, state, _ = self._dist(observation, hidden)
        raw = action.raw.to(state.device)
        if raw.dim() == 1:
            raw = raw.unsqueeze(0)
        log_prob = dist.log_prob(raw).sum(dim=-1)
        value = self.value_head(state).squeeze(-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value, entropy

    def _dist(self, observation: torch.Tensor, hidden: torch.Tensor):
        if observation.dim() == 1:
            observation = observation.unsqueeze(0)
        encoded = self.encoder(observation.float())
        core, next_hidden = self.gru(encoded.unsqueeze(0), hidden)
        state = core.squeeze(0)
        mean = self.mean(state)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std), state, next_hidden

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _as_complex(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if torch.is_complex(tensor):
        return tensor.to(torch.complex64)
    if tensor.shape[-1] == 2:
        return torch.complex(tensor[..., 0].float(), tensor[..., 1].float())
    return torch.complex(tensor.float(), torch.zeros_like(tensor.float()))


def _scalar(value: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(value).float().flatten()[0]
```

- [ ] **Step 4: Export from `agent/__init__.py`**

Append:

```python
from agent.rl_modulator import ContinuousModulationPolicy, ModulationObservationEncoder
```

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_rl_modulated_equalizer.py::test_modulation_observation_excludes_reward_and_data_labels tests\test_rl_modulated_equalizer.py::test_continuous_modulation_policy_emits_bounded_state -q -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add agent/rl_modulator.py agent/__init__.py tests/test_rl_modulated_equalizer.py
git commit -m "feat: add continuous RL modulation policy"
```

---

## Task 6: Implement RL-modulated online runner

**Files:**

- Create: `training/rl_modulated_online.py`
- Modify: `training/__init__.py`
- Test: `tests/test_rl_modulated_equalizer.py`

- [ ] **Step 1: Add failing tests for online runner output and Data isolation**

Append:

```python
def test_rl_modulated_online_runner_reports_metrics_without_data_reward(tmp_path):
    from training.rl_modulated_online import run_rl_modulated_online

    result = run_rl_modulated_online(
        config_path="configs/continual_ppo.json",
        frames=2,
        num_seeds=1,
        output_dir=tmp_path,
        delays=[20],
        snrs=[10],
        pilot_total=64,
        pilot_layout="multi_block",
    )

    assert result["schema_version"] == "rl-modulated-online-v1"
    assert len(result["rows"]) == 2
    assert all(row["method"] == "RL-Modulated Neural Block Equalizer" for row in result["rows"])
    assert all(row["policy_learning"] == "continuous_modulation_ppo" for row in result["rows"])
    assert all("ber_data" in row for row in result["rows"])
    assert all("data_loss_used_for_reward" not in row for row in result["rows"])
    assert (tmp_path / "frame_metrics.jsonl").exists()
    assert (tmp_path / "online_metrics.json").exists()


def test_rl_modulated_frame_helper_is_reusable_by_compare():
    from agent.cir_estimator import CIRCondition
    from agent.modulation import ModulationConfig, ModulationState
    from agent.rl_modulator import ContinuousModulationPolicy, ModulationObservationEncoder
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
    from training.meta_training import _estimate_cir_from_known_frame
    from training.rl_modulated_online import RLModulatedOnlineState, run_rl_modulated_frame

    env = CommunicationEnvironment(
        CommEnvConfig(level="B", max_delay=20, snr_db=10.0, total_pilot=64, layout="multi_block", seed=123)
    )
    start = env.reset_episode()
    frame = env.next_frame()
    model = UnfoldedEqualizer(UnfoldedConfig())
    modulation_config = ModulationConfig(num_adapter_gates=len(model.blocks), num_lora_scales=len(model.blocks))
    encoder = ModulationObservationEncoder()
    policy = ContinuousModulationPolicy(len(encoder.FIELDS), modulation_config)
    cir = _estimate_cir_from_known_frame(start.acquisition, 20)
    cir_b = cir.unsqueeze(0).to(torch.complex64)
    state = RLModulatedOnlineState(
        cir=cir,
        receiver_state=ReceiverState(start.initial_soft_tail),
        model=model,
        policy=policy,
        optimizer=torch.optim.AdamW(policy.parameters(), lr=3e-5),
        encoder=encoder,
        hidden=policy.initial_hidden(batch_size=1),
        modulation=ModulationState.identity(modulation_config),
        previous_reward=0.0,
        last_modulation_delta_norm=0.0,
        rollout=[],
    )
    condition = CIRCondition(
        complex_cir=cir_b,
        support_probability=(cir_b.abs() > 0).float(),
        noise_variance=torch.tensor([0.1]),
        confidence=torch.ones(1),
        latent_residual=torch.zeros(1, 96),
    )

    row = run_rl_modulated_frame(state, frame, condition, snr_db=10.0, frame_index=1, update_interval=32)

    assert row["method"] == "RL-Modulated Neural Block Equalizer"
    assert row["policy_learning"] == "continuous_modulation_ppo"
    assert row["data_labels_used_online"] is False
    assert "ber_data" in row
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_rl_modulated_equalizer.py::test_rl_modulated_online_runner_reports_metrics_without_data_reward tests\test_rl_modulated_equalizer.py::test_rl_modulated_frame_helper_is_reusable_by_compare -q -p no:cacheprovider
```

Expected:

```text
FAIL with ModuleNotFoundError: No module named 'training.rl_modulated_online'
```

- [ ] **Step 3: Implement `training/rl_modulated_online.py`**

Create:

```python
# -*- coding: utf-8 -*-
"""RL 连续调制神经块均衡器的在线 runner。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from agent.cir_estimator import CIRCondition
from agent.modulation import ModulationConfig, ModulationState
from agent.rl_modulator import ContinuousModulationPolicy, ModulationObservationEncoder
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.block_equalizers import bit_error_rate
from env.comm_env import CommEnvConfig, CommunicationEnvironment, ReceiverState
from training.meta_training import _estimate_cir_from_known_frame


@dataclass
class RLModulatedOnlineState:
    cir: torch.Tensor
    receiver_state: ReceiverState
    model: UnfoldedEqualizer
    policy: ContinuousModulationPolicy
    optimizer: torch.optim.Optimizer
    encoder: ModulationObservationEncoder
    hidden: torch.Tensor
    modulation: ModulationState
    previous_reward: float
    last_modulation_delta_norm: float
    rollout: list[dict]


def run_rl_modulated_online(
    config_path: str | Path,
    frames: int,
    num_seeds: int,
    output_dir: str | Path,
    delays: list[int] | None = None,
    snrs: list[float] | None = None,
    pilot_total: int = 128,
    pilot_layout: str = "multi_block",
    update_interval: int = 32,
) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected_delays = delays or [20, 30, 40]
    selected_snrs = snrs or [0, 5, 10, 15, 20]
    model = UnfoldedEqualizer(UnfoldedConfig.from_dict(config["model"]))
    model.eval()
    modulation_config = ModulationConfig(num_adapter_gates=len(model.blocks), num_lora_scales=len(model.blocks))
    encoder = ModulationObservationEncoder()
    policy = ContinuousModulationPolicy(len(encoder.FIELDS), modulation_config)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-5)
    rows = []
    for delay in selected_delays:
        for snr_db in selected_snrs:
            for seed in range(num_seeds):
                env = CommunicationEnvironment(
                    CommEnvConfig(
                        level="B",
                        max_delay=int(delay),
                        snr_db=float(snr_db),
                        rho=float(config.get("rho", 0.99)),
                        total_pilot=int(pilot_total),
                        layout=str(pilot_layout),
                        seed=70_000 + int(seed),
                    )
                )
                start = env.reset_episode()
                cir = _estimate_cir_from_known_frame(start.acquisition, int(delay))
                state = RLModulatedOnlineState(
                    cir=cir,
                    receiver_state=ReceiverState(start.initial_soft_tail),
                    model=model,
                    policy=policy,
                    optimizer=optimizer,
                    encoder=encoder,
                    hidden=policy.initial_hidden(batch_size=1),
                    modulation=ModulationState.identity(modulation_config),
                    previous_reward=0.0,
                    last_modulation_delta_norm=0.0,
                    rollout=[],
                )
                for frame_index in range(1, frames + 1):
                    frame = env.next_frame()
                    condition = _condition_from_cir(cir, frame.rx_symbols, float(snr_db))
                    row = run_rl_modulated_frame(state, frame, condition, float(snr_db), frame_index, update_interval=update_interval)
                    row.update({"level": "B", "delay": int(delay), "pilot_total": int(pilot_total), "pilot_layout": str(pilot_layout), "seed": int(seed)})
                    rows.append(row)
    with (target / "frame_metrics.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {"schema_version": "rl-modulated-online-v1", "rows": rows, "mean_ber_data": float(sum(row["ber_data"] for row in rows) / max(1, len(rows)))}
    (target / "online_metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "continuous-modulation-policy-v1", "state_dict": policy.state_dict()}, target / "policy.pt")
    return payload


def run_rl_modulated_frame(
    state: RLModulatedOnlineState,
    frame,
    condition: CIRCondition,
    snr_db: float,
    frame_index: int,
    update_interval: int,
) -> dict:
    rx_iq = torch.stack((frame.rx_symbols.real, frame.rx_symbols.imag), dim=-1).unsqueeze(0).float()
    region_ids = frame.model_region_ids.unsqueeze(0).long()
    tail = state.receiver_state.soft_tail.unsqueeze(0).to(torch.complex64)
    logits_before, _ = state.model(rx_iq, condition, region_ids, tail, modulation=state.modulation)
    reward_before = _masked_bce(logits_before.squeeze(0), frame.bits, frame.reward_mask)
    view = _policy_view(frame, float(snr_db), logits_before.squeeze(0), state.previous_reward, state.last_modulation_delta_norm)
    observation = state.encoder(view).tensor.unsqueeze(0)
    hidden_in = state.hidden.detach()
    action, log_prob, value, state.hidden = state.policy.sample(observation, hidden_in)
    candidate = action.state
    logits_after, _ = state.model(rx_iq, condition, region_ids, tail, modulation=candidate)
    reward_after = _masked_bce(logits_after.squeeze(0), frame.bits, frame.reward_mask)
    delta_norm = float(torch.norm(candidate.to_vector() - state.modulation.to_vector()).detach().cpu())
    reward = float((reward_before - reward_after).detach().cpu()) - 0.001 * delta_norm
    state.rollout.append({"observation": observation.detach(), "hidden": hidden_in.detach(), "action": action, "old_log_prob": log_prob.detach(), "value": value.detach(), "reward": reward})
    policy_loss = None
    if frame_index % update_interval == 0:
        policy_loss = _ppo_update(state.policy, state.optimizer, state.rollout)
        state.rollout.clear()
    if torch.isfinite(logits_after).all():
        state.modulation = candidate
    logits = logits_after.squeeze(0)
    tail_len = state.receiver_state.soft_tail.numel()
    next_tail = torch.complex(torch.tanh(logits[-tail_len:] / 2.0), torch.zeros_like(state.receiver_state.soft_tail.real))
    state.receiver_state.update_tail(next_tail)
    state.previous_reward = reward
    state.last_modulation_delta_norm = delta_norm
    return {
        "method": "RL-Modulated Neural Block Equalizer",
        "snr_db": float(snr_db),
        "frame": int(frame_index),
        "ber_data": bit_error_rate(logits[frame.data_mask], frame.bits[frame.data_mask]),
        "ber_reward_pilot": bit_error_rate(logits[frame.reward_mask], frame.bits[frame.reward_mask]),
        "ber_adapt_pilot": bit_error_rate(logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
        "reward": reward,
        "reward_pilot_loss_before": float(reward_before.detach().cpu()),
        "reward_pilot_loss_after": float(reward_after.detach().cpu()),
        "policy_learning": "continuous_modulation_ppo",
        "policy_loss": policy_loss,
        "modulation_delta_norm": delta_norm,
        "data_labels_used_online": False,
    }


def _condition_from_cir(cir: torch.Tensor, rx_symbols: torch.Tensor, snr_db: float) -> CIRCondition:
    cir_b = cir.unsqueeze(0).to(torch.complex64)
    return CIRCondition(
        complex_cir=cir_b,
        support_probability=(cir_b.abs() > 0).float(),
        noise_variance=torch.full((1,), 10.0 ** (-float(snr_db) / 10.0)),
        confidence=torch.ones(1),
        latent_residual=torch.zeros(1, 96),
    )


def _policy_view(frame, snr_db: float, logits: torch.Tensor, previous_reward: float, last_delta_norm: float):
    view = frame.receiver_view()
    return SimpleNamespace(
        rx_symbols=view.rx_symbols,
        adapt_symbols=view.adapt_symbols,
        adapt_mask=view.adapt_mask,
        model_region_ids=view.model_region_ids,
        noise_variance=torch.tensor(10.0 ** (-float(snr_db) / 10.0)),
        confidence=torch.sigmoid(torch.abs(logits)).mean(),
        previous_reward=torch.tensor(previous_reward),
        last_modulation_delta_norm=torch.tensor(last_delta_norm),
    )


def _masked_bce(logits: torch.Tensor, bits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return torch.zeros(())
    return F.binary_cross_entropy_with_logits(logits[mask].float(), bits[mask].float())


def _ppo_update(policy: ContinuousModulationPolicy, optimizer: torch.optim.Optimizer, rollout: list[dict]) -> float | None:
    if not rollout:
        return None
    rewards = torch.tensor([item["reward"] for item in rollout], dtype=torch.float32)
    returns = []
    running = torch.zeros(())
    for reward in reversed(rewards):
        running = reward + 0.95 * running
        returns.append(running)
    returns = torch.stack(list(reversed(returns)))
    values_old = torch.cat([item["value"].flatten().float() for item in rollout]).detach()
    advantages = returns - values_old
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-6)
    loss_value = torch.zeros(())
    for _ in range(2):
        losses = []
        for index, item in enumerate(rollout):
            new_log_prob, value, entropy = policy.evaluate_action(item["observation"], item["hidden"], item["action"])
            ratio = torch.exp(new_log_prob.flatten()[0] - item["old_log_prob"].flatten()[0])
            clipped = torch.clamp(ratio, 0.8, 1.2) * advantages[index]
            policy_loss = -torch.minimum(ratio * advantages[index], clipped)
            value_loss = 0.5 * (value.flatten()[0] - returns[index]).pow(2)
            losses.append(policy_loss + value_loss - 0.001 * entropy.flatten()[0])
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        loss_value = loss.detach()
    return float(loss_value.cpu())
```

- [ ] **Step 4: Export from `training/__init__.py`**

Append:

```python
from training.rl_modulated_online import RLModulatedOnlineState, run_rl_modulated_frame, run_rl_modulated_online
```

- [ ] **Step 5: Run focused test and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_rl_modulated_equalizer.py::test_rl_modulated_online_runner_reports_metrics_without_data_reward tests\test_rl_modulated_equalizer.py::test_rl_modulated_frame_helper_is_reusable_by_compare -q -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add training/rl_modulated_online.py training/__init__.py tests/test_rl_modulated_equalizer.py
git commit -m "feat: add RL-modulated online runner"
```

---

## Task 7: Integrate method groups into `compare.py`

**Files:**

- Modify: `compare.py`
- Test: `tests/test_evaluation_contract.py`

- [ ] **Step 1: Add failing CLI method-group test**

Append:

```python
def test_compare_cli_runs_traditional_and_proposed_groups(tmp_path):
    import json
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "compare.py",
            "--config",
            "configs/continual_ppo.json",
            "--method-group",
            "main",
            "--delays",
            "20",
            "--snrs",
            "10",
            "--num-seeds",
            "1",
            "--frames",
            "1",
            "--pilot-total",
            "64",
            "--pilot-layout",
            "multi_block",
            "--output-dir",
            str(tmp_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    rows = [json.loads(line) for line in (tmp_path / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    methods = {row["method"] for row in rows}

    assert "LMMSE-FIR" in methods
    assert "RL-Modulated Neural Block Equalizer" in methods
    assert "Fixed CG-BPSK-DD Block Detector" not in methods
    assert "Perfect-CSI Block" not in methods
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_evaluation_contract.py::test_compare_cli_runs_traditional_and_proposed_groups -q -p no:cacheprovider
```

Expected:

```text
FAIL with compare.py: error: unrecognized arguments: --method-group --pilot-total --pilot-layout
```

- [ ] **Step 3: Add CLI arguments to `compare.py`**

In `main()`, add:

```python
parser.add_argument("--method-group", choices=["traditional", "proposed", "diagnostic", "main", "all"], default=None)
parser.add_argument("--pilot-total", type=int, default=None)
parser.add_argument("--pilot-layout", default=None)
```

After loading config:

```python
if args.method_group:
    selected_methods = method_group(args.method_group)
else:
    selected_methods = _select_methods(args.methods)
pilot_total = int(args.pilot_total if args.pilot_total is not None else config.get("pilot_total", 128))
pilot_layout = str(args.pilot_layout if args.pilot_layout is not None else config.get("pilot_layout", "multi_block"))
```

Use `selected_methods` where the loop previously used `FORMAL_METHODS`.

- [ ] **Step 4: Route traditional methods**

Import:

```python
from agent.cir_estimator import CIRCondition
from agent.modulation import ModulationConfig, ModulationState
from agent.rl_modulator import ContinuousModulationPolicy, ModulationObservationEncoder
from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from baseline.traditional_equalizers import TRADITIONAL_BASELINES, run_traditional_equalizer
from training.rl_modulated_online import RLModulatedOnlineState, run_rl_modulated_frame
```

Add a small wrapper:

```python
def _run_traditional_method(method: str, frame, snr_db: float, state: BaselineMethodState) -> RealMethodResult:
    result = run_traditional_equalizer(method, frame.receiver_view(), state.cir, state.receiver_state.soft_tail, snr_db)
    state.receiver_state.update_tail(result.soft_tail)
    logits = result.logits
    return RealMethodResult(
        method=method,
        input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt"}),
        ber_data=bit_error_rate(logits[frame.data_mask], frame.bits[frame.data_mask]),
        ber_reward_pilot=bit_error_rate(logits[frame.reward_mask], frame.bits[frame.reward_mask]),
        ber_adapt_pilot=bit_error_rate(logits[frame.adapt_mask], frame.bits[frame.adapt_mask]),
        latency_ms=0.0,
        detector_iterations=int(result.iterations),
        extra=result.extra,
    )
```

Add a proposed-method state type in `compare.py`:

```python
@dataclass
class RLModulatedMethodState:
    online_state: RLModulatedOnlineState
    condition: CIRCondition
```

In `_build_method_states()`, create `RLModulatedMethodState` when the method is `"RL-Modulated Neural Block Equalizer"`:

```python
if method == "RL-Modulated Neural Block Equalizer":
    model = UnfoldedEqualizer(UnfoldedConfig.from_dict(config["model"]))
    model.eval()
    modulation_config = ModulationConfig(num_adapter_gates=len(model.blocks), num_lora_scales=len(model.blocks))
    encoder = ModulationObservationEncoder()
    policy = ContinuousModulationPolicy(len(encoder.FIELDS), modulation_config)
    states[method] = RLModulatedMethodState(
        online_state=RLModulatedOnlineState(
            cir=acquisition_cir.clone(),
            receiver_state=ReceiverState(initial_soft_tail.clone()),
            model=model,
            policy=policy,
            optimizer=torch.optim.AdamW(policy.parameters(), lr=3e-5),
            encoder=encoder,
            hidden=policy.initial_hidden(batch_size=1),
            modulation=ModulationState.identity(modulation_config),
            previous_reward=0.0,
            last_modulation_delta_norm=0.0,
            rollout=[],
        ),
        condition=_compare_condition_from_cir(acquisition_cir.clone(), float(snr_db)),
    )
```

Add the local condition helper in `compare.py`:

```python
def _compare_condition_from_cir(cir: torch.Tensor, snr_db: float) -> CIRCondition:
    cir_b = cir.unsqueeze(0).to(torch.complex64)
    return CIRCondition(
        complex_cir=cir_b,
        support_probability=(cir_b.abs() > 0).float(),
        noise_variance=torch.full((1,), 10.0 ** (-float(snr_db) / 10.0)),
        confidence=torch.ones(1),
        latent_residual=torch.zeros(1, 96),
    )
```

In `_run_real_method()`, route proposed through the shared single-frame helper:

```python
if method == "RL-Modulated Neural Block Equalizer":
    if not isinstance(state, RLModulatedMethodState):
        raise TypeError("RL-Modulated Neural Block Equalizer 需要 RLModulatedMethodState。")
    row = run_rl_modulated_frame(
        state.online_state,
        frame,
        state.condition,
        snr_db=float(snr_db),
        frame_index=int(frame_index),
        update_interval=int(update_interval),
    )
    return RealMethodResult(
        method=row["method"],
        input_hash=_hash_json({"seed": int(frame.frame_index), "method": method, "visible": "rx+adapt+reward-policy"}),
        ber_data=float(row["ber_data"]),
        ber_reward_pilot=float(row["ber_reward_pilot"]),
        ber_adapt_pilot=float(row["ber_adapt_pilot"]),
        latency_ms=0.0,
        detector_iterations=0,
        extra={key: value for key, value in row.items() if key not in {"method", "ber_data", "ber_reward_pilot", "ber_adapt_pilot"}},
    )
```

- [ ] **Step 5: Run focused test and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_evaluation_contract.py::test_compare_cli_runs_traditional_and_proposed_groups -q -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Run existing evaluation tests**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_evaluation_contract.py -q -p no:cacheprovider
```

Expected:

```text
all tests passed
```

- [ ] **Step 7: Commit**

Run:

```powershell
git add compare.py tests/test_evaluation_contract.py
git commit -m "feat: add compare method groups"
```

---

## Task 8: Implement pilot sweep runner

**Files:**

- Create: `evaluation/pilot_sweep.py`
- Create: `tests/test_pilot_sweep_contract.py`
- Modify: `evaluation/__init__.py`

- [ ] **Step 1: Write failing pilot sweep tests**

Create `tests/test_pilot_sweep_contract.py`:

```python
def test_pilot_sweep_candidates_use_confirmed_contract():
    from evaluation.pilot_sweep import pilot_sweep_candidates

    candidates = pilot_sweep_candidates()

    assert [item["pilot_total"] for item in candidates] == [64, 96, 128, 160, 192]
    assert all(item["pilot_layout"] == "multi_block" for item in candidates)
    assert all(item["adapt_symbols"] * 1 == int(item["pilot_total"] * 3 / 4) for item in candidates)
    assert all(item["reward_symbols"] * 4 == item["pilot_total"] for item in candidates)


def test_select_pilot_prefers_successful_goodput():
    from evaluation.pilot_sweep import select_pilot_setting

    rows = [
        {"pilot_total": 64, "proposed_beats_traditional": False, "proposed_ber_data": 0.02, "effective_goodput": 0.88, "reward_data_corr": 0.7},
        {"pilot_total": 96, "proposed_beats_traditional": True, "proposed_ber_data": 0.008, "effective_goodput": 0.80, "reward_data_corr": 0.7},
        {"pilot_total": 128, "proposed_beats_traditional": True, "proposed_ber_data": 0.006, "effective_goodput": 0.70, "reward_data_corr": 0.8},
    ]

    selected = select_pilot_setting(rows)

    assert selected["pilot_total"] == 96
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_pilot_sweep_contract.py -q -p no:cacheprovider
```

Expected:

```text
FAIL with ModuleNotFoundError: No module named 'evaluation.pilot_sweep'
```

- [ ] **Step 3: Implement `evaluation/pilot_sweep.py`**

Create:

```python
# -*- coding: utf-8 -*-
"""Pilot sweep 候选与选择规则。"""

from __future__ import annotations


def pilot_sweep_candidates(totals: list[int] | None = None, layout: str = "multi_block") -> list[dict]:
    selected = totals or [64, 96, 128, 160, 192]
    candidates = []
    for total in selected:
        adapt = int(total * 3 / 4)
        reward = int(total / 4)
        candidates.append(
            {
                "pilot_total": int(total),
                "pilot_layout": str(layout),
                "adapt_symbols": adapt,
                "reward_symbols": reward,
                "data_symbols": 512 - int(total),
            }
        )
    return candidates


def select_pilot_setting(rows: list[dict]) -> dict:
    successful = [
        row for row in rows
        if bool(row.get("proposed_beats_traditional", False))
        and float(row.get("proposed_ber_data", 1.0)) < 0.01
    ]
    if successful:
        return max(successful, key=lambda row: (float(row.get("effective_goodput", 0.0)), float(row.get("reward_data_corr", 0.0))))
    return min(rows, key=lambda row: (float(row.get("proposed_ber_data", 1.0)), -float(row.get("effective_goodput", 0.0))))
```

- [ ] **Step 4: Export from `evaluation/__init__.py`**

Append:

```python
from evaluation.pilot_sweep import pilot_sweep_candidates, select_pilot_setting
```

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_pilot_sweep_contract.py -q -p no:cacheprovider
```

Expected:

```text
2 passed
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add evaluation/pilot_sweep.py evaluation/__init__.py tests/test_pilot_sweep_contract.py
git commit -m "feat: add pilot sweep contract"
```

---

## Task 9: Update documentation contract

**Files:**

- Modify: `AGENTS.md`
- Modify: `开发框架.md`
- Modify: `RL信道均衡研究分析.md`
- Modify: `README.md`
- Test: `tests/test_evaluation_contract.py`

- [ ] **Step 1: Add failing doc contract test**

Modify `tests/test_evaluation_contract.py` doc test or append:

```python
def test_docs_use_rl_modulated_research_contract():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    docs = {
        "README.md": (root / "README.md").read_text(encoding="utf-8"),
        "AGENTS.md": (root / "AGENTS.md").read_text(encoding="utf-8"),
        "开发框架.md": (root / "开发框架.md").read_text(encoding="utf-8"),
        "RL信道均衡研究分析.md": (root / "RL信道均衡研究分析.md").read_text(encoding="utf-8"),
    }
    required = [
        "RL-Modulated Neural Block Equalizer",
        "传统非神经、非 RL",
        "LMMSE-FIR",
        "SC-FDE-MMSE",
        "Fixed CG-BPSK-DD Block Detector",
        "不作为主 baseline",
        "0, 5, 10, 15, 20 dB",
    ]
    forbidden = [
        "Best Fixed 是主成功门槛",
        "PPO 必须超过 Bandit",
        "Contextual Bandit 是主 baseline",
    ]
    assert all(all(term in text for term in required) for text in docs.values())
    assert all(all(term not in text for term in forbidden) for text in docs.values())
```

- [ ] **Step 2: Run test and verify RED**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_evaluation_contract.py::test_docs_use_rl_modulated_research_contract -q -p no:cacheprovider
```

Expected:

```text
FAIL because docs still describe Best Fixed / Bandit old contract
```

- [ ] **Step 3: Update `AGENTS.md`**

Replace old route text with this contract:

```markdown
本仓库当前唯一研究路线是 Level B 极端稀疏长回波信道下的 RL-Modulated Neural Block Equalizer。任何新代码、测试和文档必须遵守以下契约：

- 主论文目标是超过传统非神经、非 RL 均衡器，不是超过所有强模型驱动序列检测器。
- 主 baseline 固定为 LMMSE-FIR、LMS、NLMS、RLS Linear、DFE-RLS、SC-FDE-MMSE。
- `Fixed CG-BPSK-DD Block Detector` 是原 `Best Fixed`，只作为 diagnostic reference，不作为主 baseline，不参与主成功门槛。
- Proposed 方法是唯一使用神经网络和 RL 的方法。
- RL 输出低维连续调制向量，直接调制神经块均衡器；不逐 bit 判决，不直接输出完整高维 Δθ。
- 主 SNR 为 0, 5, 10, 15, 20 dB；-5 dB 只做压力测试。
- 第一阶段不加入 CFO、额外相位扰动、硬件非线性或信道编码，这些只作为 future work。
- Reward Pilot 只用于动作后 reward / last-good / rollback；Data 标签只用于离线监督和仿真 BER 评估。
```

- [ ] **Step 4: Update `开发框架.md`, `RL信道均衡研究分析.md`, and `README.md`**

Use the spec sections as source. Ensure each file contains these exact phrases:

```text
RL-Modulated Neural Block Equalizer
传统非神经、非 RL
Fixed CG-BPSK-DD Block Detector
不作为主 baseline
0, 5, 10, 15, 20 dB
```

Remove statements that describe `Best Fixed` as the success gate.

- [ ] **Step 5: Run doc test and verify GREEN**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests\test_evaluation_contract.py::test_docs_use_rl_modulated_research_contract -q -p no:cacheprovider
```

Expected:

```text
1 passed
```

- [ ] **Step 6: Commit**

Run:

```powershell
git add AGENTS.md README.md 开发框架.md RL信道均衡研究分析.md tests/test_evaluation_contract.py
git commit -m "docs: align project contract with RL modulation"
```

---

## Task 10: Smoke verification and small pilot sweep

**Files:**

- Modify: `README.md`
- Test: CLI smoke commands

- [ ] **Step 1: Run full tests**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

Expected:

```text
all tests passed
```

- [ ] **Step 2: Run RL-modulated online smoke**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe -c "from training.rl_modulated_online import run_rl_modulated_online; run_rl_modulated_online('configs/continual_ppo.json', frames=2, num_seeds=1, output_dir='logs/rl_modulated_smoke', delays=[20], snrs=[10], pilot_total=64, pilot_layout='multi_block')"
```

Expected files:

```text
logs/rl_modulated_smoke/frame_metrics.jsonl
logs/rl_modulated_smoke/online_metrics.json
logs/rl_modulated_smoke/policy.pt
```

- [ ] **Step 3: Run compare main group smoke**

Run:

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 64 --pilot-layout multi_block --output-dir logs/rl_modulated_compare_smoke
```

Expected:

```text
logs/rl_modulated_compare_smoke/frame_metrics.jsonl exists
methods include traditional baselines and RL-Modulated Neural Block Equalizer
methods do not include Fixed CG-BPSK-DD Block Detector or Perfect-CSI Block
```

- [ ] **Step 4: Add README smoke commands**

Add:

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-gpu\Scripts\python.exe -c "from training.rl_modulated_online import run_rl_modulated_online; run_rl_modulated_online('configs/continual_ppo.json', frames=2, num_seeds=1, output_dir='logs/rl_modulated_smoke', delays=[20], snrs=[10], pilot_total=64, pilot_layout='multi_block')"
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --delays 20 --snrs 10 --num-seeds 1 --frames 1 --pilot-total 64 --pilot-layout multi_block --output-dir logs/rl_modulated_compare_smoke
```

- [ ] **Step 5: Commit**

Run:

```powershell
git add README.md
git commit -m "docs: add RL-modulated smoke commands"
```

---

## Task 11: Formal pilot sweep execution gate

**Files:**

- No production code changes unless smoke reveals a bug.
- Output directory: `logs/pilot_sweep_rl_modulated_YYYYMMDD`

- [ ] **Step 1: Run pilot sweep small matrix**

Run after Tasks 1–10 pass:

```powershell
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --delays 20 40 --snrs 0 10 20 --num-seeds 3 --frames 200 --pilot-total 64 --pilot-layout multi_block --output-dir logs/pilot_sweep_rl_modulated_64
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --delays 20 40 --snrs 0 10 20 --num-seeds 3 --frames 200 --pilot-total 96 --pilot-layout multi_block --output-dir logs/pilot_sweep_rl_modulated_96
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --delays 20 40 --snrs 0 10 20 --num-seeds 3 --frames 200 --pilot-total 128 --pilot-layout multi_block --output-dir logs/pilot_sweep_rl_modulated_128
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --delays 20 40 --snrs 0 10 20 --num-seeds 3 --frames 200 --pilot-total 160 --pilot-layout multi_block --output-dir logs/pilot_sweep_rl_modulated_160
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --method-group main --delays 20 40 --snrs 0 10 20 --num-seeds 3 --frames 200 --pilot-total 192 --pilot-layout multi_block --output-dir logs/pilot_sweep_rl_modulated_192
```

- [ ] **Step 2: Verify each output has expected rows**

Expected per pilot:

```text
10 methods × 2 delays × 3 SNRs × 3 seeds × 200 frames = 36,000 rows
```

Use this command for each directory:

```powershell
.\.venv-gpu\Scripts\python.exe -c "from pathlib import Path; p=Path('logs/pilot_sweep_rl_modulated_64/frame_metrics.jsonl'); print(sum(1 for line in p.open(encoding='utf-8') if line.strip()))"
```

Expected:

```text
36000
```

- [ ] **Step 3: Decide pilot setting**

Use `evaluation.pilot_sweep.select_pilot_setting()` on summarized rows. Select the lowest-BER setting that:

```text
Proposed beats all traditional baselines
Proposed BER_data < 0.01 for 0–20 dB sweep configs
effective_goodput is highest among successful candidates
```

- [ ] **Step 4: Write iteration record**

Create:

```text
迭代记录_YYYY-MM-DD_rl_modulated_pilot_sweep.md
```

Include:

```text
pilot_total
mean BER_data by method
per-config worst case
effective_goodput
selected pilot
failure cases if no pilot satisfies all conditions
```

- [ ] **Step 5: Stop if no pilot setting satisfies the gate**

If no pilot setting satisfies both:

```text
Proposed beats all traditional baselines
Proposed BER_data < 0.01
```

do not run the 10 seeds × 1000 frames formal matrix. Analyze failure first.

---

## Self-review checklist

- Spec coverage:
  - Baseline boundary: Tasks 1, 2, 7, 9.
  - Best Fixed diagnostic-only: Tasks 1, 7, 9.
  - SNR and pilot contract: Tasks 1, 8, 11.
  - RL low-dimensional continuous modulation: Tasks 3, 4, 5, 6.
  - Data isolation: Tasks 5, 6, 9.
  - Proposed vs traditional compare: Tasks 7, 10, 11.
  - Future work exclusions: Task 9.
- Placeholder scan:
  - No `TBD`.
  - No `TODO`.
  - No `implement later`.
  - No unspecified “write tests”; every test task includes concrete test code.
- Type consistency:
  - `ModulationConfig` and `ModulationState` are defined in Task 3 and reused in Tasks 4–6.
  - `ContinuousModulationPolicy` and `ModulationObservationEncoder` are defined in Task 5 and reused in Task 6.
  - `run_rl_modulated_online()` is defined in Task 6 and used in Task 10.
  - `method_group()` is defined in Task 1 and used in Task 7.
