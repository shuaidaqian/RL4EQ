# -*- coding: utf-8 -*-
"""验证统一入口复现冻结校准轨迹、信息边界与协议隔离。"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

import compare
from baseline.calibration import (
    CALIBRATED_METHODS, load_frozen_traditional_calibration,
    run_calibrated_traditional_equalizer,
)
from baseline.traditional_equalizers import estimate_acquisition_cir_with_cfo
from env.comm_env import CommunicationEnvironment
from env.experiment_config import build_comm_env_config


PROTOCOL = Path("configs/traditional_calibration_20261101.json")
CONFIG = Path("configs/eme_long_memory_v2_head_transfer.json")


@pytest.mark.parametrize("snr", [0, 5, 10, 15])
def test_unified_compare_reproduces_archived_calibration_trajectory(monkeypatch, snr):
    """与接入前已经完成的独立核验记录对照，覆盖真实 CIR 估计和跨帧历史。"""
    monkeypatch.setenv("RL4EQ_LINEAR_OPERATOR_BACKEND", "fft")
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    fixture = json.loads(Path("tests/fixtures/traditional_calibration_trace_20261101.json").read_text(encoding="utf-8"))
    calibration = load_frozen_traditional_calibration(PROTOCOL)
    env = CommunicationEnvironment(build_comm_env_config(
        config, level="B", max_delay=116, snr_db=snr, seed=fixture["seed"],
        total_pilot=256, reward_pilot_total=32, pilot_layout="prefix", state_split="heldout_edge",
    ))
    calibration.validate_environment(env.config)
    start = env.reset_episode()
    cir, cfo = estimate_acquisition_cir_with_cfo(start.acquisition, 116)
    methods = tuple(CALIBRATED_METHODS)
    states = compare._build_method_states(
        methods, start.initial_soft_tail, cir, config, delay=116, snr_db=snr, seed=1,
        acquisition_cfo=cfo, traditional_calibration=calibration,
    )
    for index in range(3):
        frame = env.next_frame()
        for method in methods:
            result = compare._run_new_or_single_method(method, frame, snr, states[method], 116, index + 1, 4)
            expected = next(row for row in fixture["rows"] if row["snr_db"] == snr
                            and row["frame"] == index and row["kind"] == CALIBRATED_METHODS[method])
            assert result.ber_data == pytest.approx(expected["ber_data"], abs=1e-9)
            assert result.extra["traditional_calibration_sha256"] == calibration.sha256
            assert result.extra["uses_neural_network"] is False
            assert result.extra["uses_rl"] is False


def test_protocol_checks_actual_channel_and_pilot_contract():
    calibration = load_frozen_traditional_calibration(PROTOCOL)
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    env = build_comm_env_config(config, level="B", max_delay=116, snr_db=10, seed=47001,
                                total_pilot=256, reward_pilot_total=32, pilot_layout="prefix",
                                state_split="heldout_edge")
    calibration.validate_environment(env)
    for changed in (replace(env, total_pilot=128), replace(env, reward_pilot_total=64),
                    replace(env, acquisition_to_data_gap_seconds=30), replace(env, rho=0.8)):
        with pytest.raises(ValueError, match="不匹配"):
            calibration.validate_environment(changed)
    with pytest.raises(ValueError, match="重叠"):
        calibration.validate_environment(replace(env, seed=36000))


@pytest.mark.parametrize("lookahead", [0, 116])
def test_calibrated_receiver_does_not_read_hidden_labels_or_symbols(monkeypatch, lookahead):
    """传统推理可读取接收信号，但隐藏发送符号不能进入同步或均衡。"""
    monkeypatch.setenv("RL4EQ_LINEAR_OPERATOR_BACKEND", "fft")
    calibration = load_frozen_traditional_calibration(PROTOCOL)
    env = CommunicationEnvironment(build_comm_env_config(
        calibration.source_config, level="B", max_delay=116, snr_db=10, seed=47002,
        total_pilot=256, reward_pilot_total=32, pilot_layout="prefix", state_split="heldout_edge",
    ))
    start = env.reset_episode()
    cir, cfo = estimate_acquisition_cir_with_cfo(start.acquisition, 116)
    frame = env.next_frame(lookahead_len=lookahead)
    hidden = ~frame.adapt_mask
    bits, symbols = frame.bits.clone(), frame.tx_symbols.clone()
    bits[hidden], symbols[hidden] = ~bits[hidden], complex(float("nan"), float("nan"))
    changed = replace(frame, bits=bits, tx_symbols=symbols)
    for method in CALIBRATED_METHODS:
        outputs = [run_calibrated_traditional_equalizer(
            method, item.receiver_view(), cir, start.initial_soft_tail, 10,
            acquisition_cfo=cfo, calibration=calibration,
        ) for item in (frame, changed)]
        torch.testing.assert_close(outputs[0].logits, outputs[1].logits, atol=0, rtol=0)
        assert outputs[0].logits.shape == frame.rx_symbols.shape
        assert outputs[0].soft_tail.shape == start.initial_soft_tail.shape


def test_calibrated_compare_cli_audits_shared_rx_and_rejects_mixed_resume(tmp_path):
    output = tmp_path / "compare"
    command = [sys.executable, "compare.py", "--config", str(CONFIG), "--methods", *CALIBRATED_METHODS,
               "--traditional-calibration", str(PROTOCOL), "--snrs", "10", "--seed-start", "47003",
               "--num-seeds", "1", "--frames", "1", "--state-split", "heldout_edge",
               "--device", "cpu", "--lookahead-len", "116", "--output-dir", str(output)]
    completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", timeout=90)
    assert completed.returncode == 0, completed.stderr
    rows = [json.loads(line) for line in (output / "frame_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 3
    assert len({row["rx_sha256"] for row in rows}) == 1
    assert len({row["lookahead_rx_sha256"] for row in rows}) == 1
    assert all(row["lookahead_len"] == 116 and len(row["lookahead_rx_sha256"]) == 64 for row in rows)
    assert all(row["lookahead_frame_index"] == row["frame"] for row in rows)
    assert all(len(row["rx_sha256"]) == 64 and row["environment_seed"] == 97003 for row in rows)
    before = (output / "frame_metrics.jsonl").read_bytes()
    changed = list(command)
    position = changed.index("--traditional-calibration")
    del changed[position:position + 2]
    failed = subprocess.run(changed + ["--resume"], capture_output=True, timeout=90)
    assert failed.returncode != 0
    assert (output / "frame_metrics.jsonl").read_bytes() == before
    changed_window = list(command)
    changed_window[changed_window.index("--lookahead-len") + 1] = "0"
    failed = subprocess.run(changed_window + ["--resume"], capture_output=True, timeout=90)
    assert failed.returncode != 0
    assert (output / "frame_metrics.jsonl").read_bytes() == before


def test_cross_frame_calibration_and_frozen_entry_use_identical_windows(tmp_path, monkeypatch):
    """校准与部署封装必须同窗、同补偿、同当前帧 tail；新协议不能退回零后缀。"""
    import hashlib
    from baseline.calibration import FrozenTraditionalCalibration
    from scripts.calibrate_linear_baselines import run_calibration

    monkeypatch.setenv("RL4EQ_LINEAR_OPERATOR_BACKEND", "fft")
    original = load_frozen_traditional_calibration(PROTOCOL)
    config = {**original.source_config, "receiver_lookahead_len": 116, "main_snrs": [10]}
    selection = {**original.selection,
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()}
    calibration = FrozenTraditionalCalibration(config, selection, "x" * 64, "test")
    result = run_calibration(config, [47127], 2, tmp_path / "calibration.json", selection=selection)
    env = CommunicationEnvironment(build_comm_env_config(
        config, level="B", max_delay=116, snr_db=10, seed=47127, total_pilot=256,
        reward_pilot_total=32, pilot_layout="prefix", state_split="offline_train"))
    start = env.reset_episode()
    cir, cfo = estimate_acquisition_cir_with_cfo(start.acquisition, 116)
    tails = {method: start.initial_soft_tail.clone() for method in CALIBRATED_METHODS}
    for index in range(2):
        frame = env.next_frame(lookahead_len=116)
        for method, kind in CALIBRATED_METHODS.items():
            output = run_calibrated_traditional_equalizer(method, frame.receiver_view(), cir, tails[method],
                10, acquisition_cfo=cfo, calibration=calibration)
            tails[method] = output.soft_tail
            ber = float(((output.logits[frame.data_mask] >= 0) != frame.bits[frame.data_mask]).float().mean())
            row = next(row for row in result["rows"] if row["frame"] == index and row["kind"] == kind)
            assert row["ber_data"] == ber
            assert row["lookahead_len"] == 116
        with pytest.raises(ValueError, match="校准窗口不匹配"):
            run_calibrated_traditional_equalizer("LMMSE-FIR",
                replace(frame, lookahead_rx_symbols=None, lookahead_adapt_symbols=None,
                        lookahead_frame_index=None).receiver_view(), cir, start.initial_soft_tail,
                10, acquisition_cfo=cfo, calibration=calibration)
