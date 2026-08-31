# -*- coding: utf-8 -*-
"""验证 acquisition 与数据开始之间的信道老化建模。"""

import torch

from env.comm_env import CommEnvConfig, CommunicationEnvironment
from agent.cir_estimator import pilot_sparse_cir_update


def _config(gap_seconds: float) -> CommEnvConfig:
    return CommEnvConfig(
        level="B",
        max_delay=116,
        snr_db=10.0,
        rho=0.9991470306520146,
        total_pilot=128,
        layout="prefix",
        seed=9021,
        profile_name="eme_long_memory_v2",
        sample_rate_hz=10_000.0,
        symbol_rate_hz=10_000.0,
        frame_len=1024,
        max_delay_seconds=0.0116,
        coherence_time_seconds=120.0,
        strong_path_count=(12, 24),
        diffuse_energy_ratio=(0.20, 0.35),
        acquisition_to_data_gap_seconds=gap_seconds,
    )


def test_acquisition_data_gap_increases_first_data_cir_mismatch():
    immediate = CommunicationEnvironment(_config(0.0))
    aged = CommunicationEnvironment(_config(60.0))
    immediate_start = immediate.reset_episode()
    aged_start = aged.reset_episode()

    immediate_first = immediate.next_frame().true_cir
    aged_first = aged.next_frame().true_cir
    assert immediate_first is not None
    assert aged_first is not None

    immediate_error = torch.sum(torch.abs(immediate_first - immediate_start.acquisition.true_cir) ** 2)
    aged_error = torch.sum(torch.abs(aged_first - aged_start.acquisition.true_cir) ** 2)
    assert aged_error > immediate_error * 5.0


def test_acquisition_data_gap_is_recorded_in_state_metadata():
    environment = CommunicationEnvironment(_config(30.0))
    environment.reset_episode()

    metadata = environment.state_metadata()
    assert metadata["acquisition_to_data_gap_seconds"] == 30.0
    assert metadata["coherence_time_seconds"] == 120.0


def test_pilot_sparse_cir_update_uses_only_prefix_pilot_and_reduces_cir_error():
    environment = CommunicationEnvironment(_config(60.0))
    start = environment.reset_episode()
    frame = environment.next_frame()
    assert frame.true_cir is not None

    updated = pilot_sparse_cir_update(
        frame,
        start.acquisition.true_cir,
        frame.tail_symbols,
        max_paths=24,
        alpha=1.0,
    )
    previous_error = torch.sum(torch.abs(start.acquisition.true_cir - frame.true_cir) ** 2)
    updated_error = torch.sum(torch.abs(updated - frame.true_cir) ** 2)
    assert updated_error < previous_error
