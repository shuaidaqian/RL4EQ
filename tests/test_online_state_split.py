import copy
import json
from pathlib import Path

import pytest

from env.online_state import load_online_state_split


def _experiment():
    return {
        "profile_name": "eme_long_memory_v2",
        "channel_profile": "eme_long_memory_v2",
        "max_delay_seconds": 0.0116,
        "strong_path_count": [12, 24],
        "diffuse_energy_ratio": [0.2, 0.35],
        "online_state_splits": {
            "offline_train": {
                "strong_path_count": [12, 18],
                "diffuse_energy_ratio": [0.20, 0.28],
                "cfo_abs_range": [0.0002, 0.0006],
                "phase_noise_std_range": [0.0002, 0.0006],
            },
            "heldout_edge": {
                "strong_path_count": [19, 24],
                "diffuse_energy_ratio": [0.29, 0.35],
                "cfo_abs_range": [0.0006, 0.0008],
                "phase_noise_std_range": [0.0006, 0.0008],
            },
            "drift": {
                "strong_path_count": [12, 24],
                "diffuse_energy_ratio": [0.20, 0.35],
                "cfo_abs_range": [0.0002, 0.0008],
                "phase_noise_std_range": [0.0002, 0.0008],
            },
        },
    }


def test_load_online_state_split_keeps_profile_contract_and_normalizes_ranges():
    split = load_online_state_split(_experiment(), "heldout_edge")

    assert split.name == "heldout_edge"
    assert split.profile_name == "eme_long_memory_v2"
    assert split.max_delay_seconds == pytest.approx(0.0116)
    assert split.strong_path_count == (19, 24)
    assert split.diffuse_energy_ratio == pytest.approx((0.29, 0.35))
    assert split.cfo_abs_range == pytest.approx((0.0006, 0.0008))
    assert split.phase_noise_std_range == pytest.approx((0.0006, 0.0008))


def test_online_state_sample_is_seed_reproducible_and_within_declared_ranges():
    split = load_online_state_split(_experiment(), "heldout_edge")

    first = split.sample(17)
    second = split.sample(17)
    different = split.sample(18)

    assert first == second
    assert first != different
    assert 19 <= first["strong_path_count"] <= 24
    assert 0.29 <= first["diffuse_energy_ratio"] <= 0.35
    assert 0.0006 <= first["cfo_abs"] <= 0.0008
    assert 0.0006 <= first["phase_noise_std"] <= 0.0008


@pytest.mark.parametrize("split_name", ["offline_train", "heldout_edge", "drift"])
def test_state_split_cannot_change_fixed_physical_profile(split_name):
    experiment = _experiment()
    split = load_online_state_split(experiment, split_name)

    assert split.profile_name == experiment["profile_name"]
    assert split.max_delay_seconds == pytest.approx(experiment["max_delay_seconds"])
    assert split.pilot_layout == "prefix"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["online_state_splits"]["heldout_edge"].update(
            {"max_delay_seconds": 0.01}
        ),
        lambda value: value["online_state_splits"]["heldout_edge"].update(
            {"pilot_layout": "two_block"}
        ),
        lambda value: value["online_state_splits"]["heldout_edge"].update(
            {"strong_path_count": [25, 26]}
        ),
    ],
)
def test_invalid_state_split_is_rejected_without_mutating_experiment(mutation):
    experiment = _experiment()
    original = copy.deepcopy(experiment)
    mutation(experiment)

    with pytest.raises(ValueError):
        load_online_state_split(experiment, "heldout_edge")

    assert experiment["profile_name"] == original["profile_name"]
    assert experiment["max_delay_seconds"] == original["max_delay_seconds"]


def test_eme_environment_uses_selected_split_without_exposing_true_state():
    from env.comm_env import CommunicationEnvironment
    from env.experiment_config import build_comm_env_config

    experiment = json.loads(
        (Path(__file__).resolve().parents[1] / "configs" / "eme_long_memory_v2.json").read_text(
            encoding="utf-8"
        )
    )
    experiment["online_state_splits"] = {
        "heldout_edge": {
            "strong_path_count": [19, 24],
            "diffuse_energy_ratio": [0.29, 0.35],
            "cfo_abs_range": [0.0006, 0.0008],
            "phase_noise_std_range": [0.0006, 0.0008],
        }
    }
    config = build_comm_env_config(
        experiment,
        level="B",
        snr_db=10.0,
        seed=7017,
        max_delay=116,
        total_pilot=128,
        pilot_layout="prefix",
        state_split="heldout_edge",
    )

    assert config.state_split == "heldout_edge"
    assert config.max_delay == 116
    assert config.layout == "prefix"
    environment = CommunicationEnvironment(config)
    environment.reset_episode()
    state = environment.state_metadata()
    frame = environment.next_frame()

    assert state["state_split"] == "heldout_edge"
    assert 19 <= state["strong_path_count"] <= 24
    assert 0.29 <= state["diffuse_energy_ratio"] <= 0.35
    assert 0.0006 <= abs(state["cfo_cycles_per_symbol"]) <= 0.0008
    assert 0.0006 <= state["phase_noise_std"] <= 0.0008
    view = frame.receiver_view()
    assert not hasattr(view, "true_cir")
    assert not hasattr(view, "cfo_cycles_per_symbol")
