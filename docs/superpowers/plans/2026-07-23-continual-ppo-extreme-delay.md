# Continual PPO 极端长时延均衡实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从 `main@69b4064` 构建可在 GTX 1650 4 GB 上训练的物理引导长时延块均衡器，并实现部署期间持续更新的分层 recurrent PPO，使 Level B 九个主配置的未编码 BPSK `BER_data` 分别低于 0.01。

**Architecture:** 接收机使用已知 warm-up、acquisition frame、多连续 Pilot 子块和跨帧 soft-tail；Hybrid CIR Estimator 输出显式稀疏复 CIR 与 latent residual，Unfolded Equalizer 使用 `H/H^H` 迭代检测。离线依次完成信道校准、CIR训练、Perfect-CIR检测、Estimated-CIR联合训练和 first-order meta-training；随后 Continual Hierarchical GRU PPO 以 Reward Pilot 即时/累计改善为 reward，每 32 帧在线更新 policy。

**Tech Stack:** Python 3.12、PyTorch 2.11+cu128、NumPy、Matplotlib、Pytest；Windows PowerShell；单卡 NVIDIA GTX 1650 4 GB。

---

## 实施纪律

严格按 Task 1→16 执行。Task 2–10 的阶段门槛未通过时，不开始正式 PPO。每个生产代码变更遵循 Red→Green→Refactor；每个任务独立提交。设计依据为 `docs/superpowers/specs/2026-07-23-continual-ppo-extreme-delay-design.md`。

```text
Perfect-CSI 强检测器：Level B 九配置分别具备 BER < 0.01 可达性
Perfect-CIR Unfolded：Level B 九配置分别 BER < 0.01
Best Fixed Adaptation：Level B 九配置分别 BER < 0.1
Reward/Data Spearman：>= 0.6
Continual PPO：Level B 九配置分别 BER < 0.01
```

## 目标文件结构

```text
agent/adaptation_controller.py    混合动作执行、安全投影和分状态回滚
agent/cir_estimator.py             显式 CIR/support/noise/confidence/latent
agent/continual_policy.py          结构化编码器、GRU、混合 Actor-Critic
agent/peft.py                      Adapter、LoRA、参数组和状态快照
agent/unfolded_equalizer.py        H/H^H 展开式 BPSK 均衡器
baseline/block_equalizers.py       Perfect/Pilot-estimated 块均衡与迭代检测
baseline/channel_tracking.py       OMP/稀疏 LS、Kalman/RLS CIR 跟踪
baseline/legacy_equalizers.py      简单 LMMSE-FIR、DFE-RLS 参考
env/channel_profiles.py            Level A/B/C 配置、分类和频谱指标
env/comm_env.py                    acquisition、普通帧和 episode 状态
env/extreme_delay_channel.py       分层长回波、漂移和跨帧 ISI
env/frame_structure.py             64/96/128/160 与三种布局
env/linear_operator.py             可微复卷积 H/H^H
evaluation/bootstrap.py            seed/episode block bootstrap
evaluation/metrics.py              BER、CIR、goodput、相关性和延迟
evaluation/runner.py               分片、增量写出、resume 和聚合
training/checkpointing.py           模型/优化器/RNG/阶段状态
training/curriculum.py             Stage 1–4 课程监督训练
training/meta_training.py          first-order episodic meta-training
training/continual_ppo.py          rollout、GAE、混合 PPO、在线 KL 回滚
tests/test_channel_protocol.py
tests/test_receiver_architecture.py
tests/test_meta_adaptation.py
tests/test_continual_ppo.py
tests/test_evaluation_contract.py
calibrate_channel.py
pretrain.py
online_train.py
compare.py
```

---

### Task 1: 从 main 清理旧路线并建立新项目契约

**Files:**
- Delete: `CLAUDE.md`, `agent/actor_critic.py`, `agent/ppo.py`
- Delete: `env/channel_models.py`, `env/ldpc_coding.py`
- Delete: tracked `__pycache__/`, `*.pyc`, `logs/*.png`
- Delete: `reference/2102.00178-*`, `reference/2603.02489-*`
- Create: `.gitignore`, package `__init__.py`, `tests/test_evaluation_contract.py`
- Create: `requirements-gpu.txt`, `scripts/setup_gpu_env.ps1`, `scripts/clean_local_artifacts.ps1`
- Create: `calibrate_channel.py`, `pretrain.py`, `compare.py`
- Replace: `online_train.py`

- [ ] **Step 1: 写仓库边界失败测试**

```python
from pathlib import Path


def test_repository_contains_only_continual_ppo_entrypoints():
    root = Path(__file__).resolve().parents[1]
    required = {"calibrate_channel.py", "pretrain.py", "online_train.py", "compare.py"}
    obsolete = {"CLAUDE.md", "agent/actor_critic.py", "agent/ppo.py", "env/channel_models.py", "env/ldpc_coding.py"}
    assert all((root / path).exists() for path in required)
    assert all(not (root / path).exists() for path in obsolete)
```

- [ ] **Step 2: 运行测试并确认因旧文件/新入口而失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_evaluation_contract.py -q -p no:cacheprovider`
Expected: FAIL，指出入口缺失或旧文件存在。

- [ ] **Step 3: 清理并创建可执行的新入口壳**

入口壳统一提供可运行的 CLI 版本信息：

```python
# -*- coding: utf-8 -*-
"""RL4EQ Continual PPO 新路线入口。"""

import argparse

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()
    if args.version:
        print("RL4EQ continual-ppo schema-v1")


if __name__ == "__main__":
    main()
```

`.gitignore` 必须包含 `.venv*/`、`logs/`、`pretrained/`、`artifacts/`、`tmp/`、`__pycache__/`、`.pytest_cache/`、`*.pyc`。

- [ ] **Step 4: 清理旧本地产物并创建全新 GPU 环境**

`scripts/clean_local_artifacts.ps1` 只允许删除仓库根目录下精确命名的 `.git_original/`、`.pytest_cache/`、`.agents/`、所有 `__pycache__/`、旧 `logs/` 和旧 `pretrained/`；每个目标先 `Resolve-Path` 并验证以仓库绝对路径开头，明确跳过 `.git/`。`requirements-gpu.txt` 固定 NumPy、SciPy、Matplotlib、Pytest 的兼容版本；`scripts/setup_gpu_env.ps1 -Recreate` 用同样路径守卫删除旧 `.venv-gpu/`，再依次执行 `py -3.12 -m venv .venv-gpu`、升级 pip、从 PyTorch cu128 index 安装 torch，并安装其余依赖。PyTorch wheel 自带 CUDA runtime，不修改系统 CUDA Toolkit。

Run: `powershell -ExecutionPolicy Bypass -File scripts/clean_local_artifacts.ps1`
Run: `powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_env.ps1 -Recreate`
Run: `.\.venv-gpu\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"`
Expected: 输出 CUDA 版 torch、CUDA runtime 和 `NVIDIA GeForce GTX 1650`。

- [ ] **Step 5: 运行测试并确认通过**

Expected: `1 passed`。

- [ ] **Step 6: 提交**

```powershell
git add -A
git commit -m "refactor: reset main to continual PPO research baseline"
```

---

### Task 2: 实现 Level A/B/C 信道配置与校准指标

**Files:**
- Create: `env/channel_profiles.py`, `configs/channel_levels.json`
- Create/Modify: `tests/test_channel_protocol.py`

- [ ] **Step 1: 写 Level B 约束与种子复现失败测试**

```python
def test_level_b_profile_obeys_power_constraints_and_seed():
    cfg = ChannelProfileConfig(level=ChannelLevel.B, max_delay=40, seed=17)
    first = sample_profile(cfg)
    second = sample_profile(cfg)
    assert first.delays == second.delays
    assert np.allclose(first.taps, second.taps)
    assert first.delays[0] == 0 and first.delays[-1] == 40
    assert 3 <= len(first.delays) <= 7
    assert 0.0 <= first.strongest_gap_db <= 6.0
    assert first.max_delay_relative_db >= -10.0
    assert 0.35 <= first.delayed_energy_ratio <= 0.75
    assert np.isclose(np.sum(np.abs(first.taps) ** 2), 1.0, atol=1e-6)
```

- [ ] **Step 2: 运行定点测试并验证 `ModuleNotFoundError`**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_channel_protocol.py::test_level_b_profile_obeys_power_constraints_and_seed -q -p no:cacheprovider`
Expected: FAIL，错误包含 `No module named 'env.channel_profiles'`。

- [ ] **Step 3: 实现公开 API 和受约束拒绝采样**

```python
class ChannelLevel(str, Enum):
    A = "A"
    B = "B"
    C = "C"


@dataclass(frozen=True)
class ChannelProfileConfig:
    level: ChannelLevel
    max_delay: int
    seed: int
    spectral_grid: int = 2048


@dataclass(frozen=True)
class ChannelProfile:
    delays: Sequence[int]
    taps: np.ndarray
    strongest_gap_db: float
    max_delay_relative_db: float
    delayed_energy_ratio: float
    notch_depth_db: float
    condition_proxy: float
```

频谱指标使用 2048 点 `sum(tap*exp(-j*w*delay))`；采样最多拒绝 10,000 次，超限抛出包含 level/seed 的异常，禁止静默放宽。

- [ ] **Step 4: 增加 Level A/C、无效配置和频谱指标测试**

Level A 测试固定验证 3–5 路、最强/次强差 6–15 dB、最大时延 tap 为 -10 至 -20 dB、延迟能量 10%–35%；Level C 验证 3–10 路、延迟能量 50%–90%，并确保 Level C 标签不会进入 Level B 汇总。无效 level、重复 delay 和拒绝采样超限必须抛出 `ValueError`/`ProfileSamplingError`。

- [ ] **Step 5: 运行并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_channel_protocol.py -q -p no:cacheprovider
git add env/channel_profiles.py configs/channel_levels.json tests/test_channel_protocol.py
git commit -m "feat: add calibrated level A B C channel profiles"
```

---

### Task 3: 实现分层长回波、固定 Es/N0 与跨帧 ISI

**Files:**
- Create: `env/extreme_delay_channel.py`
- Modify: `tests/test_channel_protocol.py`

- [ ] **Step 1: 写 SNR、ISI、support 固定和功率归一化失败测试**

```python
def test_channel_uses_known_history_and_preserves_support():
    channel = ExtremeDelayChannel(ExtremeDelayChannelConfig(level="B", max_delay=20, snr_db=10.0, rho=0.99, seed=3))
    channel.reset_episode(torch.ones(20, dtype=torch.complex64))
    first_delays = channel.delays
    channel.transmit(torch.ones(64, dtype=torch.complex64), add_noise=False)
    second = channel.transmit(torch.zeros(64, dtype=torch.complex64), add_noise=False)
    assert torch.any(second[:20].abs() > 0)
    assert channel.delays == first_delays
    assert torch.allclose(channel.tap_power.sum(), torch.tensor(1.0), atol=1e-5)
```

- [ ] **Step 2: 运行定点测试并验证失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_channel_protocol.py::test_channel_uses_known_history_and_preserves_support -q -p no:cacheprovider`
Expected: FAIL，错误包含 `No module named 'env.extreme_delay_channel'`。

- [ ] **Step 3: 实现状态机**

```python
@dataclass(frozen=True)
class ExtremeDelayChannelConfig:
    level: str = "B"
    max_delay: int = 40
    snr_db: float = 10.0
    rho: float = 0.99
    seed: int = 42


class ExtremeDelayChannel:
    def reset_episode(self, known_warmup: torch.Tensor) -> None:
        """采样 episode 基准 tap 并设置已知发送历史。"""
        self._initialize_state(known_warmup)

    def transmit(self, symbols: torch.Tensor, add_noise: bool = True) -> torch.Tensor:
        """施加卷积、跨帧历史、增益漂移和固定 Es/N0 噪声。"""
        return self._transmit_frame(symbols, add_noise)

    def true_cir(self) -> torch.Tensor:
        """仅向离线监督与 Perfect-CSI 诊断暴露真实 CIR。"""
        return self._current_cir.clone()
```

生产实现使用固定 `noise_variance=10**(-snr_db/10)`，I/Q 方差各为一半，不得按当前输出功率缩放。tap 按 `h_t=rho*h_(t-1)+(1-rho)*h_base+sqrt(1-rho**2)*innovation` 围绕 `base_taps` 均值回归，support 不变，每帧归一化；第一版不允许路径出生/消失。

- [ ] **Step 4: 运行协议全测并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_channel_protocol.py -q -p no:cacheprovider
git add env/extreme_delay_channel.py tests/test_channel_protocol.py
git commit -m "feat: implement stateful level-aware extreme delay channel"
```

---

### Task 4: 实现多块 Pilot、Acquisition 与 Soft-Tail 环境

**Files:**
- Replace: `env/frame_structure.py`, `env/comm_env.py`
- Modify: `tests/test_channel_protocol.py`

- [ ] **Step 1: 写 12 种结构、Reward 隐藏和 acquisition 失败测试**

```python
@pytest.mark.parametrize("total", [64, 96, 128, 160])
@pytest.mark.parametrize("layout", ["prefix", "two_block", "multi_block"])
def test_frame_masks_and_unknown_regions(total, layout):
    frame = FrameGenerator(FrameConfig(total_pilot=total, layout=layout, max_delay=40), seed=9).generate(2)
    assert int(frame.adapt_mask.sum()) == 3 * total // 4
    assert int(frame.reward_mask.sum()) == total // 4
    assert int(frame.data_mask.sum()) == 512 - total
    assert not torch.any(frame.adapt_mask & frame.reward_mask)
    assert torch.equal(frame.unknown_region_mask, frame.reward_mask | frame.data_mask)
    assert torch.all(frame.model_region_ids[frame.reward_mask] == frame.model_region_ids[frame.data_mask][0])
    assert max(frame.adapt_block_lengths) >= 48
```

- [ ] **Step 2: 运行 12 组合定点测试并验证旧接口失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_channel_protocol.py::test_frame_masks_and_unknown_regions -q -p no:cacheprovider`
Expected: 12 个参数组合均 FAIL，缺少 `FrameGenerator` 或新 mask。

- [ ] **Step 3: 实现帧/环境 API**

```python
@dataclass(frozen=True)
class FrameConfig:
    frame_len: int = 512
    total_pilot: int = 128
    layout: str = "multi_block"
    max_delay: int = 40


@dataclass
class EpisodeStart:
    warmup_symbols: torch.Tensor
    acquisition: "ReceivedFrame"
    initial_soft_tail: torch.Tensor
```

`reset_episode()` 返回已知 warm-up、全已知 acquisition 和精确初始 tail；`receiver_view()` 删除 bits、reward mask和true CIR；offline view 才暴露监督标签。

- [ ] **Step 4: 测试 first-frame tail、不同方法状态隔离和标签边界**

新增测试断言第一普通帧 tail 等于 acquisition 最后 `max_delay` 个已知符号；两个 receiver state 更新其中一个不会改变另一个；`receiver_view` 不含 Reward mask/位置/标签、Data 标签或 true CIR；Pilot PN 在相同 seed/frame 可复现、跨帧变化。

- [ ] **Step 5: 运行并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_channel_protocol.py -q -p no:cacheprovider
git add env/frame_structure.py env/comm_env.py tests/test_channel_protocol.py
git commit -m "feat: add acquisition and distributed pilot frame protocol"
```

---

### Task 5: 实现复卷积算子与 Perfect-CSI 可达性校准

**Files:**
- Create: `env/linear_operator.py`, `baseline/block_equalizers.py`
- Replace: `calibrate_channel.py`
- Modify: `tests/test_receiver_architecture.py`

- [ ] **Step 1: 写伴随一致性和无噪声恢复失败测试**

```python
def test_linear_operator_adjoint_identity():
    operator = LinearChannelOperator(frame_len=64, max_delay=7)
    x = torch.randn(2, 64, dtype=torch.complex64)
    y = torch.randn(2, 64, dtype=torch.complex64)
    cir = torch.randn(2, 8, dtype=torch.complex64)
    tail = torch.zeros(2, 7, dtype=torch.complex64)
    lhs = (operator.forward(x, cir, tail).conj() * y).sum()
    rhs = (x.conj() * operator.adjoint(y, cir)).sum()
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


def test_perfect_csi_detector_recovers_noiseless_bpsk():
    result = perfect_csi_cg_detect(rx, true_cir, known_tail, noise_variance, iterations=32)
    assert bit_error_rate(result.logits, bits) == 0.0
```

- [ ] **Step 2: 运行定点测试并确认模块缺失**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py -q -p no:cacheprovider`
Expected: FAIL，缺少 `LinearChannelOperator` 和 `perfect_csi_cg_detect`。

- [ ] **Step 3: 实现统一 H/H^H、CG 与解析 BPSK 迭代器**

`LinearChannelOperator.forward()` 必须显式拼接每种方法独立维护的 `soft_tail`，`adjoint()` 只返回当前帧对应梯度；`perfect_csi_cg_detect()` 解 `(H^H H + sigma2 I)x=H^H y` 后输出 BPSK LLR；`iterative_bpsk_detect()` 在同一算子上执行阻尼软判决。所有 baseline 只接收 acquisition/Adapt Pilot 估计或 true CIR 的明确接口，禁止读取 Reward/Data 标签。

```python
@dataclass(frozen=True)
class DetectionResult:
    logits: torch.Tensor
    probabilities: torch.Tensor
    soft_tail: torch.Tensor
    iterations: int


def perfect_csi_cg_detect(
    rx: torch.Tensor,
    cir: torch.Tensor,
    soft_tail: torch.Tensor,
    noise_variance: torch.Tensor,
    iterations: int,
) -> DetectionResult:
    """用已知 CIR 求解整帧块检测，仅用于可达性和上界基线。"""
```

- [ ] **Step 4: 实现 10,000 候选校准入口和冻结清单**

`calibrate_channel.py` 接收 `--candidates 10000 --seeds configs/eval_seeds.json --output artifacts/calibration`，逐候选写 JSONL，汇总九个 Level B 主配置的 BER、2048 点最小频谱增益和条件数 proxy，并把最终 A/B/C 阈值写入 `configs/channel_levels.json`。若任一主配置 Perfect-CSI BER 不低于 0.01，入口以退出码 2 结束，不启动神经训练。

- [ ] **Step 5: 运行测试、smoke 校准并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py tests/test_channel_protocol.py -q -p no:cacheprovider
.\.venv-gpu\Scripts\python.exe calibrate_channel.py --candidates 32 --frames-per-config 2 --output artifacts/calibration_smoke
git add env/linear_operator.py baseline/block_equalizers.py calibrate_channel.py tests/test_receiver_architecture.py
git commit -m "feat: add physical channel operators and reachability calibration"
```

---

### Task 6: 实现 Hybrid Sparse CIR Estimator

**Files:**
- Create: `agent/cir_estimator.py`, `baseline/channel_tracking.py`
- Modify: `tests/test_receiver_architecture.py`

- [ ] **Step 1: 写输出契约、因果信息边界与稀疏估计失败测试**

```python
def test_cir_estimator_contract_and_label_isolation():
    out_a = estimator(rx_iq, adapt_symbols, adapt_mask, unknown_region_ids)
    changed_reward = reward_bits.logical_not()
    out_b = estimator(rx_iq, adapt_symbols, adapt_mask, unknown_region_ids)
    assert out_a.complex_cir.shape == (2, 41)
    assert torch.is_complex(out_a.complex_cir)
    assert out_a.support_probability.shape == (2, 41)
    assert out_a.latent_residual.shape == (2, 96)
    assert torch.equal(out_a.complex_cir, out_b.complex_cir)
    assert changed_reward.shape == reward_bits.shape
```

- [ ] **Step 2: 确认测试因公开类型缺失而失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py::test_cir_estimator_contract_and_label_isolation -q -p no:cacheprovider`
Expected: FAIL，错误包含 `No module named 'agent.cir_estimator'`。

- [ ] **Step 3: 实现解析初始化与神经残差校正**

```python
@dataclass
class CIRCondition:
    complex_cir: torch.Tensor
    support_probability: torch.Tensor
    noise_variance: torch.Tensor
    confidence: torch.Tensor
    latent_residual: torch.Tensor


class HybridCIREstimator(nn.Module):
    """Adapt Pilot/acquisition 驱动的显式稀疏 CIR 与隐式残差估计器。"""

    def forward(
        self,
        rx_iq: torch.Tensor,
        adapt_symbols: torch.Tensor,
        adapt_mask: torch.Tensor,
        region_ids: torch.Tensor,
    ) -> CIRCondition:
        # 稀疏 LS/OMP 给出解析初值，卷积编码器只校正 CIR 和置信度。
        return self._decode(self._encode(rx_iq, adapt_symbols, adapt_mask, region_ids))
```

损失固定为 `complex_cir_mse + 0.2*support_bce + 0.1*noise_log_mse + 0.05*latent_l2`；acquisition 可更新 tracker 状态，普通帧只使用 Adapt Pilot。`baseline/channel_tracking.py` 提供相同输入边界的 Sparse-LS、Kalman 和 RLS 跟踪器。

- [ ] **Step 4: 增加 CIR NMSE、support F1、Reward/Data 标签扰动测试**

测试要求无噪声合成信道的 CIR NMSE `< -20 dB`；改变 Reward/Data bits 但保持模型可见张量不变时，`CIRCondition` 逐元素相同。

- [ ] **Step 5: 运行并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py -q -p no:cacheprovider
git add agent/cir_estimator.py baseline/channel_tracking.py tests/test_receiver_architecture.py
git commit -m "feat: add hybrid sparse CIR estimator"
```

---

### Task 7: 实现 Physics-Guided Unfolded Equalizer 与 PEFT 参数组

**Files:**
- Create: `agent/peft.py`, `agent/unfolded_equalizer.py`
- Modify: `tests/test_receiver_architecture.py`

- [ ] **Step 1: 写整帧输出、非因果依赖与 PEFT 冻结失败测试**

```python
def test_unfolded_equalizer_is_noncausal_and_peft_is_bounded():
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=4))
    logits_a, _ = model(rx_iq, condition, region_ids, soft_tail)
    rx_changed = rx_iq.clone()
    rx_changed[:, -1] += 3.0
    logits_b, _ = model(rx_changed, condition, region_ids, soft_tail)
    assert logits_a.shape == (2, 512)
    assert not torch.equal(logits_a[:, 100], logits_b[:, 100])
    model.set_trainable_groups({"adapter", "attention_lora"})
    assert model.trainable_parameter_count() <= 0.10 * model.parameter_count()
```

- [ ] **Step 2: 确认测试因模型不存在而失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py::test_unfolded_equalizer_is_noncausal_and_peft_is_bounded -q -p no:cacheprovider`
Expected: FAIL，错误包含 `UnfoldedEqualizer` 未定义。

- [ ] **Step 3: 实现展开迭代、FiLM、Adapter 和 LoRA**

每层严格执行：

```python
residual = rx_complex - operator.forward(soft_symbols, condition.complex_cir)
gradient = operator.adjoint(residual, condition.complex_cir)
proposal = soft_symbols + alpha[layer] * gradient
features = torch.stack((proposal.real, proposal.imag, residual.real, residual.imag), dim=-1)
features = film(denoiser(features, condition.latent_residual), condition)
soft_symbols = (1.0 - damping[layer]) * torch.tanh(head(features)) + damping[layer] * soft_symbols
```

`PEFTRegistry` 仅注册 `conditioner_film`、`adapter`、`attention_lora`、`ffn_lora`、`head` 五组，提供 `snapshot(groups)`、`restore(snapshot)`、`delta_norm(snapshot)` 和参数占比检查。默认总模型不超过 2M 参数；线上组合优先不超过 5%，任何动作硬上限 10%。

跨模块统一使用以下序列化值：底层注册组为 `conditioner_film`、`adapter`、`attention_lora`、`ffn_lora`、`head`；策略可选值为前四个单组以及 `adapter_lora`、`conditioner_peft`。`adapter_lora` 展开为 Adapter + 两类 LoRA + head，`conditioner_peft` 再加入 Conditioner/FiLM；组合在 `PEFTRegistry.resolve()` 中展开，policy、controller、checkpoint 和日志不得另造别名。

- [ ] **Step 4: 写 strict checkpoint、快照恢复和泄漏测试**

保存 `model_config + state_dict + schema_version` 后由新实例 `strict=True` 加载；恢复 snapshot 后选定组 bitwise 相同；Reward/Data 标签翻转不得改变 forward 输入或输出。

- [ ] **Step 5: 运行、统计规模并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_receiver_architecture.py -q -p no:cacheprovider
.\.venv-gpu\Scripts\python.exe -c "from agent.unfolded_equalizer import *; m=UnfoldedEqualizer(UnfoldedConfig()); print(sum(p.numel() for p in m.parameters()))"
git add agent/peft.py agent/unfolded_equalizer.py tests/test_receiver_architecture.py
git commit -m "feat: add physics guided unfolded equalizer and PEFT groups"
```

---

### Task 8: 实现可恢复 checkpoint 与分片实验状态

**Files:**
- Create: `training/checkpointing.py`, `evaluation/runner.py`
- Modify: `tests/test_evaluation_contract.py`

- [ ] **Step 1: 写模型/优化器/RNG 精确续跑失败测试**

```python
def test_checkpoint_resume_reproduces_next_step(tmp_path):
    uninterrupted = run_tiny_training(seed=7, steps=4)
    partial = run_tiny_training(seed=7, steps=2, save_to=tmp_path / "state.pt")
    resumed = run_tiny_training(seed=999, steps=4, resume=tmp_path / "state.pt")
    assert partial.completed_steps == 2
    assert torch.equal(uninterrupted.model_vector, resumed.model_vector)
    assert uninterrupted.next_batch_hash == resumed.next_batch_hash
```

- [ ] **Step 2: 确认续跑状态测试失败**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_evaluation_contract.py::test_checkpoint_resume_reproduces_next_step -q -p no:cacheprovider`
Expected: FAIL，错误包含 `training.checkpointing` 或 `run_tiny_training` 缺失。

- [ ] **Step 3: 实现原子 checkpoint 与 JSONL 分片**

`save_checkpoint()` 先写同目录 `.tmp`，flush/fsync 后用 Python `os.replace()` 原子替换，保存模型、optimizer、scheduler、GradScaler、Python/NumPy/Torch CPU/CUDA RNG、stage、global_step、config hash。`ShardWriter` 每帧追加 JSONL，每 32 帧 flush；`completed_keys()` 以 `(method, level, delay, snr, rho, pilot, layout, seed, frame)` 去重，`--resume` 跳过已完成键。

- [ ] **Step 4: 验证损坏 checkpoint 明确报错且不覆盖最后好状态**

- [ ] **Step 5: 运行并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_evaluation_contract.py -q -p no:cacheprovider
git add training/checkpointing.py evaluation/runner.py tests/test_evaluation_contract.py
git commit -m "feat: add deterministic resumable experiment state"
```

---

### Task 9: 实现 Stage 1–4 Pilot 条件课程监督训练

**Files:**
- Create: `training/curriculum.py`
- Replace: `pretrain.py`
- Create: `configs/continual_ppo.json`, `configs/eval_seeds.json`
- Modify: `tests/test_meta_adaptation.py`

- [ ] **Step 1: 写课程顺序、全程 Pilot 条件和选择指标失败测试**

```python
def test_curriculum_is_always_pilot_conditioned_and_level_ordered():
    schedule = build_curriculum(test_config())
    assert [phase.name for phase in schedule] == [
        "cir_level_a", "perfect_cir_level_a", "estimated_cir_level_a", "estimated_cir_level_b"
    ]
    assert all(phase.total_pilot in {64, 96, 128, 160} for phase in schedule)
    assert all(phase.layout in {"prefix", "two_block", "multi_block"} for phase in schedule)
    assert all(phase.uses_pilot_condition for phase in schedule)
```

- [ ] **Step 2: 确认旧预训练入口不满足契约**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_meta_adaptation.py::test_curriculum_is_always_pilot_conditioned_and_level_ordered -q -p no:cacheprovider`
Expected: FAIL，缺少课程 API 或旧入口没有四阶段配置。

- [ ] **Step 3: 实现四阶段 trainer 和配置校验**

Stage 1 独立训练 CIR estimator；Stage 2 用 true CIR 训练 unfolded，并在 Level B 九配置分别验证 `<0.01`；Stage 3 用 estimated CIR 在 Level A 联合训练；Stage 4 按 Level A→B、Pilot 160→128→96→64、SNR 20→15→10 dB、delay 20→30→40 的顺序逐级提高难度，最终 Level B 比例为 100%。12 种 Pilot 结构共享采样，每 batch 只实例化一种结构以控制显存。损失为 Data BCE 主项，加 Adapt/Reward BCE、CIR 多任务损失和 unfolded 中间层深监督；Reward 标签在离线监督中可进入 loss，但不得进入 receiver input。

`pretrain.py` 必须支持 `--stage`, `--steps`, `--batch-size`, `--accumulation-steps`, `--amp`, `--resume`, `--save-dir`；每 100 step 保存 `last.pt`，按 Level B 九配置验证集平均 BER 保存 `model_best.pt`，同时保存逐配置 BER，不能以平均值掩盖失败配置。

- [ ] **Step 4: 用 2-step GPU smoke 验证四阶段和 strict load**

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage all --steps 2 --batch-size 1 --accumulation-steps 1 --amp --save-dir pretrained/smoke
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_meta_adaptation.py::test_pretrained_checkpoint_strict_load -q -p no:cacheprovider
```

Expected: 四阶段均产生有限 loss，`model_best.pt` 可 strict load，显存峰值 `< 3.5 GB`。

- [ ] **Step 5: 运行测试并提交**

```powershell
git add training/curriculum.py pretrain.py configs/continual_ppo.json configs/eval_seeds.json tests/test_meta_adaptation.py
git commit -m "feat: add pilot conditioned level A to B curriculum training"
```

---

### Task 10: 实现 First-Order Episodic Meta-Training 与 Fixed Gate

**Files:**
- Create: `training/meta_training.py`
- Modify: `pretrain.py`, `tests/test_meta_adaptation.py`

- [ ] **Step 1: 写 support/query 隔离、first-order 和九配置门槛失败测试**

```python
def test_meta_episode_uses_adapt_support_and_offline_query_only():
    episode = build_meta_episode(frame)
    assert torch.equal(episode.support_mask, frame.adapt_mask)
    assert torch.equal(episode.query_mask, frame.reward_mask | frame.data_mask)
    adapted = first_order_inner_update(model, episode, groups={"adapter", "attention_lora"}, steps=2)
    assert all(parameter.grad_fn is None for parameter in adapted.fast_weights.values())
    assert episode.receiver_view.data_bits is None
```

- [ ] **Step 2: 确认 meta API 缺失**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_meta_adaptation.py::test_meta_episode_uses_adapt_support_and_offline_query_only -q -p no:cacheprovider`
Expected: FAIL，缺少 `build_meta_episode`/`first_order_inner_update`。

- [ ] **Step 3: 实现一阶 episodic meta-training**

Support 只用 Adapt Pilot 做 inner update；offline outer loss 使用 Reward Pilot + Data，且 Data 标签仅存在于 trainer 的 query loss 作用域，不传给模型、动作选择器或 observation。使用 first-order fast weights、AMP 和 gradient accumulation，不构建二阶图。每 episode 从五个 PEFT 参数组中采样合法组合，为后续控制动作提供共同初始化。

- [ ] **Step 4: 实现 Best Fixed 搜索和硬前置门槛**

在固定开发集上枚举合法参数组、`steps={1,2,4}`、`iterations={2,4,6,8}` 与离散学习率网格，且只以 Adapt 更新、Reward 选择。保存 `artifacts/fixed_gate/best_fixed.json`；九个 Level B 主配置必须分别 `<0.1`，并在开发 5 seeds 中至少 4 seeds 达标，否则 `pretrain.py --stage meta` 以退出码 3 结束，禁止启动 PPO。

- [ ] **Step 5: 运行 meta smoke、门槛单测并提交**

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage meta --steps 2 --batch-size 1 --resume pretrained/smoke/last.pt --save-dir pretrained/meta_smoke
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_meta_adaptation.py -q -p no:cacheprovider
git add training/meta_training.py pretrain.py tests/test_meta_adaptation.py
git commit -m "feat: add first order meta adaptation and fixed gate"
```

---

### Task 11: 实现指标、Reward/Data 相关性与 Pilot 结构选择

**Files:**
- Create: `evaluation/metrics.py`
- Modify: `tests/test_evaluation_contract.py`, `pretrain.py`

- [ ] **Step 1: 写 BER、effective goodput、Spearman 与逐配置聚合失败测试**

```python
def test_metrics_do_not_average_away_failed_configs():
    rows = synthetic_rows(delays=[20, 30, 40], snrs=[10, 15, 20], seeds=5)
    summary = summarize_main_matrix(rows)
    assert len(summary.per_config) == 9
    assert summary.all_below(0.01) == all(row.ber_data < 0.01 for row in summary.per_config)
    assert effective_goodput(
        ber=0.01,
        data_symbols=384,
        ordinary_frames=1000,
        warmup_symbols=40,
        acquisition_symbols=512,
        frame_symbols=512,
    ) == pytest.approx(384 * 1000 * 0.99 / (40 + 512 + 512 * 1000))
    assert spearman_reward_data([1, 2, 3], [0.2, 0.4, 0.9]).correlation == 1.0
```

- [ ] **Step 2: 确认指标模块缺失**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_evaluation_contract.py::test_metrics_do_not_average_away_failed_configs -q -p no:cacheprovider`
Expected: FAIL，错误包含 `No module named 'evaluation.metrics'`。

- [ ] **Step 3: 实现统一指标 schema**

每帧必须写出：`method, level, delay, snr_db, rho, pilot_total, pilot_layout, seed, frame, ber_data, ber_adapt_pilot, ber_reward_pilot, pilot_loss, cir_nmse_db, adapt_params, adapt_steps, detector_iterations, latency_ms, parameter_delta_norm, rollback, effective_goodput`。`generalization` 只汇总 Level C/未见配置并单列，绝不混入 Level B 主平均。

Reward/Data Spearman 使用同一动作前后 `Reward loss 改善` 与 `Data BER 改善` 的配对样本；开发集相关系数必须 `>=0.6`，否则输出 `reward_alignment_failed.json` 并阻断 PPO。

- [ ] **Step 4: 实现 12→3 的 Pilot 候选筛选流程**

共享训练初筛全部 `4×3` 结构，按九配置门槛、Reward/Data Spearman、effective goodput、最坏 seed 依次排序；只允许前 2–3 个候选精调。保存 `artifacts/pilot_selection/shortlist.json`，其中包含全部候选分数、淘汰原因和 2–3 个 PPO 候选。最终 3→1 选择必须等 Continual PPO 真实开发结果产生后在 Task 15 完成，避免在 PPO 门槛之前提前冻结布局。

- [ ] **Step 5: 运行并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_evaluation_contract.py -q -p no:cacheprovider
git add evaluation/metrics.py tests/test_evaluation_contract.py pretrain.py
git commit -m "feat: add gate aware metrics and pilot selection"
```

---

### Task 12: 实现结构化 Observation 与分层 Recurrent Mixed Policy

**Files:**
- Create: `agent/continual_policy.py`
- Modify: `tests/test_continual_ppo.py`

- [ ] **Step 1: 写 observation 泄漏、GRU 状态和混合分布失败测试**

```python
def test_policy_observation_excludes_reward_and_data_labels():
    view = receiver_view_with_hidden_labels()
    obs_a = ObservationEncoder()(view.with_hidden_labels(reward=all_zero, data=all_zero))
    obs_b = ObservationEncoder()(view.with_hidden_labels(reward=all_one, data=all_one))
    assert torch.equal(obs_a.tensor, obs_b.tensor)
    assert "reward_bits" not in obs_a.fields and "data_bits" not in obs_a.fields


def test_recurrent_policy_emits_legal_hierarchical_action():
    action, log_prob, value, hidden = policy.sample(observation, torch.zeros(1, 1, 128))
    assert action.mode in {"skip", "update-channel", "update-equalizer", "joint-update", "detector-refine", "rollback"}
    assert action.steps in {1, 2, 4}
    assert action.iterations in {2, 4, 6, 8}
    assert hidden.shape == (1, 1, 128)
    assert torch.isfinite(log_prob + value).all()
```

- [ ] **Step 2: 确认 policy 模块缺失**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_continual_ppo.py::test_recurrent_policy_emits_legal_hierarchical_action -q -p no:cacheprovider`
Expected: FAIL，错误包含 `No module named 'agent.continual_policy'`。

- [ ] **Step 3: 实现结构化编码与动作条件 mask**

Observation 由 CIR 序列编码器、Pilot/残差统计编码器、历史适配/奖励编码器组成，再送入单层 `GRU(hidden_size=128)`。禁止任何当前 Reward 标签/损失、Data 标签/BER；允许上一帧动作完成后得到的 reward、shadow loss 和安全状态。

```python
@dataclass(frozen=True)
class HierarchicalAction:
    mode: str
    parameter_group: str
    steps: int
    detector_iterations: int
    learning_rate: float
    proximal_weight: float
    reconstruction_weight: float
    damping: float
    cir_trust: float
```

离散头依次采样 mode、parameter group、steps、iterations；parameter group 只输出 Task 7 定义的 `conditioner_film/adapter/attention_lora/ffn_lora/adapter_lora/conditioner_peft`。连续头用 squash-Normal 映射到 `lr=[1e-6,1e-3]`、`proximal=[1e-6,1e-2]`、`reconstruction=[0,1]`、`damping=[0,0.95]`、`cir_trust=[0,1]`。`skip/rollback` mask 掉无意义更新参数，`detector-refine` 只控制 iterations/damping，非法组合 log-prob 不参与 PPO。

- [ ] **Step 4: 增加 no-GRU 和 no-detector-control 消融配置测试**

`ablation="no_gru"` 使用相同编码器和 MLP 替代 recurrent state；`ablation="no_detector_control"` 固定 iterations/damping 并从 log-prob 中去掉对应头。策略总参数必须 `<1M`。

- [ ] **Step 5: 运行并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_continual_ppo.py -q -p no:cacheprovider
git add agent/continual_policy.py tests/test_continual_ppo.py
git commit -m "feat: add recurrent hierarchical mixed adaptation policy"
```

---

### Task 13: 实现分层动作控制器、Shadow Reward 与安全回滚

**Files:**
- Create: `agent/adaptation_controller.py`
- Modify: `tests/test_continual_ppo.py`

- [ ] **Step 1: 写动作可达性、跨帧持续和异常回滚失败测试**

```python
def test_controller_persists_safe_updates_and_resets_between_seeds():
    controller.reset_episode(seed=1, checkpoint=checkpoint)
    before = controller.peft_vector().clone()
    result = controller.execute(joint_action, frame)
    after = controller.peft_vector().clone()
    assert not torch.equal(before, after)
    assert torch.equal(after, controller.peft_vector())
    controller.reset_episode(seed=2, checkpoint=checkpoint)
    assert torch.equal(before, controller.peft_vector())


def test_nonfinite_update_hard_rolls_back():
    result = controller.execute(action_causing_nan, frame)
    assert result.rollback and result.reward == -1.0
    assert torch.equal(controller.peft_vector(), controller.last_safe_vector())
```

- [ ] **Step 2: 确认控制器模块缺失**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_continual_ppo.py::test_controller_persists_safe_updates_and_resets_between_seeds -q -p no:cacheprovider`
Expected: FAIL，错误包含 `No module named 'agent.adaptation_controller'`。

- [ ] **Step 3: 实现动作执行、安全投影和按组件 last-safe 快照**

`update-channel` 更新 CIR conditioner/tracker；`update-equalizer` 更新动作指定 PEFT 组；`joint-update` 顺序更新 channel 与 equalizer；`detector-refine` 不做梯度，只改变展开迭代；`rollback` 恢复最近安全快照。梯度非有限、loss 非有限、参数占比超过 10% 或 delta norm 超硬阈值时立即回滚并记 `-1`。普通性能下降不自动回滚。

- [ ] **Step 4: 实现即时/累计 shadow reward 且证明不读 Data**

```python
def compute_reward(loss_before, loss_after, shadow_loss, beta, eps=1e-8):
    immediate = torch.log((loss_before + eps) / (loss_after + eps))
    cumulative = torch.log((shadow_loss + eps) / (loss_after + eps))
    return immediate + beta * cumulative
```

`loss_before/loss_after` 只来自当前 Reward Pilot，且标签仅在动作完成后的 reward evaluator 内解封；`shadow_loss` 来自 episode-start 冻结、整个 episode 都不更新的 shadow receiver 在 Reward Pilot 上的 loss。函数签名不得接受 Data tensor、BER 或标签。`no_cumulative_shadow_reward` 消融令 `beta=0`。

- [ ] **Step 5: 运行并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_continual_ppo.py -q -p no:cacheprovider
git add agent/adaptation_controller.py tests/test_continual_ppo.py
git commit -m "feat: add safe hierarchical adaptation controller and shadow reward"
```

---

### Task 14: 实现部署期间持续更新的 PPO 与在线入口

**Files:**
- Create: `training/continual_ppo.py`
- Replace: `online_train.py`
- Modify: `tests/test_continual_ppo.py`

- [ ] **Step 1: 写 prequential 时序、32 帧更新和 seed 隔离失败测试**

```python
def test_continual_ppo_updates_during_deployment_without_cross_seed_state():
    run = tiny_online_run(frames=65, update_interval=32, seed=11)
    assert run.policy_update_frames == [32, 64]
    assert run.metrics[0].measured_before_current_frame_update
    second = tiny_online_run(frames=1, update_interval=32, seed=12)
    assert second.initial_policy_hash == run.offline_policy_hash
    assert second.initial_receiver_hash == run.offline_receiver_hash
```

- [ ] **Step 2: 确认旧在线入口与持续 PPO 契约不符**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_continual_ppo.py::test_continual_ppo_updates_during_deployment_without_cross_seed_state -q -p no:cacheprovider`
Expected: FAIL，缺少 continual rollout 或 32 帧更新记录。

- [ ] **Step 3: 实现 recurrent rollout、GAE 与 mixed-action PPO loss**

Rollout 保存动作前 observation/hidden、分层离散及连续动作、联合 log-prob、value、reward、done 和合法 mask；每 32 帧计算 GAE。更新时从 rollout 起始 hidden 重放 GRU，用 clipped surrogate、value clipping、entropy bonus、gradient clipping，并对离散/连续头分别记录 KL。KL 超配置阈值时恢复 policy optimizer 前快照并降低学习率，不能影响 receiver 的安全状态。

- [ ] **Step 4: 实现 acquisition + 300/1000 帧 continual 入口**

`online_train.py` 必须读取 Pilot shortlist、meta receiver 和 fixed gate。先用仿真 episode 对每个候选做 PPO offline initialization，保存共同的 `policy_init.pt`；部署评估时每 seed 从同一 offline receiver、policy 和 optimizer 规则重置，经 1 acquisition frame 后按帧 prequential 评估并继续训练 PPO。支持 `--phase offline|continual|all`, `--frames`, `--num-seeds`, `--update-interval 32`, `--resume`, `--amp`, `--output-dir`。开发默认 300 帧，正式默认 1000 帧；不提供 Frozen PPO 运行模式，也不允许从随机初始化直接做正式评估。

- [ ] **Step 5: 运行 65-frame GPU smoke 并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_continual_ppo.py -q -p no:cacheprovider
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/meta_smoke/model_best.pt --frames 65 --num-seeds 1 --update-interval 32 --amp --output-dir logs/ppo_smoke
git add training/continual_ppo.py online_train.py tests/test_continual_ppo.py
git commit -m "feat: train recurrent PPO continually during deployment"
```

Expected: 恰好在帧 32、64 写两次 policy update；指标有限；显存峰值 `<3.5 GB`；checkpoint 可 `--resume`。

---

### Task 15: 实现公平 Baseline、正式比较与分层 Bootstrap

**Files:**
- Create: `baseline/legacy_equalizers.py`, `evaluation/bootstrap.py`
- Modify: `baseline/block_equalizers.py`, `compare.py`
- Modify: `tests/test_evaluation_contract.py`

- [ ] **Step 1: 写 baseline 信息边界、配对种子和 bootstrap 失败测试**

```python
@pytest.mark.parametrize("method", FORMAL_METHODS)
def test_all_methods_receive_same_label_free_frame(method):
    received = paired_frame(seed=23)
    result_a = run_method(method, received.hide_reward_and_data_labels())
    result_b = run_method(method, received.hide_reward_and_data_labels())
    assert result_a.input_hash == result_b.input_hash == received.observable_hash


def test_hierarchical_bootstrap_resamples_seed_then_ten_frame_blocks():
    interval = paired_block_bootstrap(rows, seed=7, repetitions=2000, block_length=10)
    assert interval.resampling_order == ("seed", "contiguous_frame_block")
    assert interval.block_length == 10
```

- [ ] **Step 2: 确认 compare 与统计接口缺失**

Run: `.\.venv-gpu\Scripts\python.exe -m pytest tests/test_evaluation_contract.py::test_hierarchical_bootstrap_resamples_seed_then_ten_frame_blocks -q -p no:cacheprovider`
Expected: FAIL，错误包含 `No module named 'evaluation.bootstrap'`。

- [ ] **Step 3: 完成正式方法集合**

固定方法为 `Perfect-CSI Block`、`Sparse CIR + Kalman/RLS`、`Block LMMSE/CG`、`DFE-RLS`、`Analytic Iterative BPSK`、`Legacy LMMSE-FIR`、`Legacy DFE`、`No Adapt`、`Best Fixed`、`Drift-Aware Pilot Rule`、`Contextual Bandit`、`Continual PPO`。除明确标注 Perfect-CSI 的可达性基线外，所有方法只能用 acquisition/Adapt Pilot；Reward Pilot 只用于动作后的 reward/评估；Data 标签只用于仿真指标。代码、CLI 和文档中均不得存在 Data Oracle。

- [ ] **Step 4: 实现正式矩阵、最小消融和报告**

`compare.py` 对 Level B `delay=20/30/40 × SNR=10/15/20 × rho=0.99` 执行 10 unseen seeds ×1000 frames；Level A、Level C 和 `rho=0.999/0.95` 单独输出。三项 PPO 消融固定为 `no_gru`、`no_cumulative_shadow_reward`、`no_detector_control`。配对 bootstrap 先重采样 seed/episode，再在 seed 内采样连续 10 帧块，输出均值、标准差、中位数、四分位数、95% CI、最坏 seed、成功 seed 比例和相对 Best Fixed/Rule/Bandit 的配对差；prequential BER 另报 1–100、101–300、301–1000 三段。

对 Task 11 shortlist 中 2–3 个结构分别完成 PPO 开发运行后，先过滤九配置 PPO BER 均 `<0.01` 且每配置至少 4/5 开发 seeds 达标的候选，再最大化含 warm-up/acquisition 开销的 effective goodput，冻结唯一 `selected.json`。正式成功判据为每配置至少 8/10 seeds 达标、至少 7/9 配置取得最低可部署 BER，且相对 Best Fixed/Rule/Bandit 配对显著改善；任一配置退化超过 0.002 必须单列，不得用整体平均掩盖。

- [ ] **Step 5: 运行 2-frame 比较 smoke 并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest tests/test_evaluation_contract.py -q -p no:cacheprovider
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --pretrained pretrained/meta_smoke/model_best.pt --policy logs/ppo_smoke/policy.pt --delays 20 40 --snrs 10 20 --num-seeds 1 --frames 2 --output-dir logs/compare_smoke
git add baseline evaluation compare.py tests/test_evaluation_contract.py
git commit -m "feat: add paired formal evaluation and required baselines"
```

Expected: 每种方法写齐统一 schema；主 Level B 与 Level C 文件分离；没有 Data Oracle 行；失败门槛逐配置列出。

---

### Task 16: 文档统一、GPU 全量验证、效果反思与带时间戳迭代记录

**Files:**
- Replace: `README.md`, `开发框架.md`, `RL信道均衡研究分析.md`, `AGENTS.md`
- Replace: `reference/README.md`, `reference/summary.txt`, `reference/references.bib`
- Create at runtime: `迭代记录_YYYY-MM-DD_HH-mm-ss.md`
- Delete: 无引用模块、重复论文文本/PDF、空目录和全部已跟踪缓存

- [ ] **Step 1: 写文档一致性和禁用概念失败测试**

```python
def test_docs_share_single_research_contract():
    docs = read_project_docs()
    required = ["Level B", "Continual PPO", "整帧缓冲", "非因果", "BER_data < 0.01", "不实现 Data Oracle"]
    forbidden = ["Data Oracle（诊断上界）", "逐符号实时输出", "LDPC 编解码实验", "CFO 补偿实验"]
    assert all(all(term in text for term in required) for text in docs.values())
    assert all(all(term not in text for term in forbidden) for text in docs.values())
```

- [ ] **Step 2: 重写四份根文档与文献索引**

文档统一说明：Level B 为主论文场景；Level C 仅压力测试；全程 Pilot 条件课程预训练；12 种 Pilot 开销/布局选择；物理展开接收机；first-order meta；Continual PPO 是第一贡献；不做 Frozen PPO 和 Data Oracle；PPO 以进一步降低 BER 为目标；只有真实重跑结果可以进入结果章节。保留 P-FTNet、在线持续学习、online Bayesian receiver、PPO reward 直接相关文献，删除 MIMO MCTS、RIS equalizer 和重复副本。

- [ ] **Step 3: 清理并执行静态质量检查**

```powershell
git ls-files | Where-Object { $_ -match '(__pycache__|\.pyc$)' } | ForEach-Object { git rm -- $_ }
rg -n "actor_critic|agent\.ppo|ldpc_coding|DataOracle|data_oracle|FrozenHierarchical" agent env baseline training evaluation *.py tests
.\.venv-gpu\Scripts\python.exe -m compileall agent env baseline training evaluation
git diff --check
```

Expected: `rg` 无命中，compileall 成功，`git diff --check` 无输出。删除仅缓存内容的 `utils/`；保留本地 `.venv-gpu/`，但由 `.gitignore` 隐藏。

- [ ] **Step 4: 执行 GPU 全量自动测试和四项 smoke**

```powershell
.\.venv-gpu\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
.\.venv-gpu\Scripts\python.exe calibrate_channel.py --candidates 10000 --frames-per-config 200 --output artifacts/calibration --resume
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage all --steps 2 --batch-size 1 --amp --save-dir pretrained/final_smoke
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/final_smoke/model_best.pt --frames 65 --num-seeds 1 --amp --output-dir logs/final_online_smoke
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --pretrained pretrained/final_smoke/model_best.pt --policy logs/final_online_smoke/policy.pt --delays 20 40 --snrs 10 20 --num-seeds 1 --frames 2 --output-dir logs/final_compare_smoke
```

- [ ] **Step 5: 分阶段运行开发规模 GPU 实验并逐门槛停检**

先运行 Stage 1–4、Pilot 精调和 meta-training；验证 Perfect-CIR 九配置 `<0.01`、Best Fixed 九配置 `<0.1`、5 seeds 至少 4 seeds 达标、Spearman `>=0.6` 后，才运行 5 seeds ×300 frames Continual PPO。任一前置门槛失败即停在对应阶段，记录证据、定位假设，并回到该阶段修正；不得用 PPO 掩盖接收机不可达。

```powershell
.\.venv-gpu\Scripts\python.exe pretrain.py --config configs/continual_ppo.json --stage all --resume pretrained/last.pt --save-dir pretrained --amp
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/model_best.pt --frames 300 --num-seeds 5 --resume --amp --output-dir logs/online_development
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --pretrained pretrained/model_best.pt --policy logs/online_development/policy.pt --num-seeds 5 --frames 200 --resume --output-dir logs/compare_development
```

- [ ] **Step 6: 达到开发门槛后执行正式 10×1000 配对矩阵**

```powershell
.\.venv-gpu\Scripts\python.exe online_train.py --config configs/continual_ppo.json --pretrained pretrained/model_best.pt --frames 1000 --num-seeds 10 --resume --amp --output-dir logs/online_formal
.\.venv-gpu\Scripts\python.exe compare.py --config configs/continual_ppo.json --pretrained pretrained/model_best.pt --policy logs/online_formal/policy.pt --num-seeds 10 --frames 1000 --resume --output-dir logs/compare_formal
```

正式运行可跨多次会话续跑；每次恢复先校验 config/checkpoint hash。GTX 1650 单候选或 PPO 开发训练若超过 12 小时，保留已完成分片并记录运行时，不擅自缩减正式统计定义。

- [ ] **Step 7: 生成带时间戳的整体迭代记录并进行预期反思**

以 Asia/Shanghai 当前时间生成 `迭代记录_YYYY-MM-DD_HH-mm-ss.md`，包含：分支基点和提交范围、GPU/CUDA/依赖、各命令及退出码、阶段耗时、九配置逐项 BER、各 baseline/消融、95% CI/最坏 seed/成功 seed 比例、Pilot 胜出原因、Spearman、显存/延迟、未通过门槛、失败修复、与预期一致/不一致之处、研究结论边界、可复现实验路径。不得把 smoke 或 Level C 结果表述为正式主结论。

- [ ] **Step 8: 最终复核整洁度并提交**

```powershell
.\.venv-gpu\Scripts\python.exe -m pytest -q -p no:cacheprovider
git status --short --ignored
git diff --check
git add -A
git commit -m "docs: finalize continual PPO research route and iteration record"
```

完成标准：全部自动测试通过；所有长入口支持 resume；Git 仅跟踪源代码、配置、文档和小型汇总，不跟踪环境/checkpoint/日志/缓存；四份根文档完全一致；迭代记录只陈述实际执行证据。
