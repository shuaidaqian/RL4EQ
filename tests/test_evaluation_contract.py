from pathlib import Path
import subprocess
import sys

import pytest
import torch

from compare import FORMAL_METHODS, paired_frame, run_method
from evaluation.bootstrap import paired_block_bootstrap
from training.checkpointing import CheckpointError, load_checkpoint, run_tiny_training
from evaluation.metrics import (
    FrameMetric,
    effective_goodput,
    select_pilot_shortlist,
    spearman_reward_data,
    summarize_main_matrix,
)


def test_repository_contains_only_continual_ppo_entrypoints():
    root = Path(__file__).resolve().parents[1]
    required = {"calibrate_channel.py", "pretrain.py", "online_train.py", "compare.py"}
    obsolete = {
        "CLAUDE.md",
        "agent/actor_critic.py",
        "agent/ppo.py",
        "env/channel_models.py",
        "env/ldpc_coding.py",
    }
    assert all((root / path).exists() for path in required)
    assert all(not (root / path).exists() for path in obsolete)


def test_repository_ignores_local_runtime_artifacts():
    root = Path(__file__).resolve().parents[1]
    ignore_file = root / ".gitignore"
    assert ignore_file.exists()
    ignored = {
        ".venv*/",
        "logs/",
        "pretrained/",
        "artifacts/",
        "tmp/",
        "__pycache__/",
        ".pytest_cache/",
        "*.pyc",
    }
    content = ignore_file.read_text(encoding="utf-8")
    assert ignored.issubset(set(content.splitlines()))


def test_gpu_setup_and_cleanup_scripts_exist():
    root = Path(__file__).resolve().parents[1]
    scripts = {"scripts/setup_gpu_env.ps1", "scripts/clean_local_artifacts.ps1"}
    assert all((root / path).exists() for path in scripts)


def test_cleanup_script_guards_repository_boundaries_and_git_directory():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "clean_local_artifacts.ps1"
    assert script.exists()
    content = script.read_text(encoding="utf-8")
    assert ".git" in content
    assert "Resolve-Path" in content
    assert "StartsWith" in content


def test_cleanup_script_removes_runtime_artifacts_without_touching_git(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init"], cwd=root, check=True, capture_output=True, text=True)
    script_dir = root / "scripts"
    script_dir.mkdir()
    source_script = Path(__file__).resolve().parents[1] / "scripts" / "clean_local_artifacts.ps1"
    target_script = script_dir / "clean_local_artifacts.ps1"
    target_script.write_text(source_script.read_text(encoding="utf-8"), encoding="utf-8")

    runtime_cache = root / "__pycache__"
    runtime_cache.mkdir()
    (runtime_cache / "stale.pyc").write_text("旧缓存", encoding="utf-8")
    logs = root / "logs"
    logs.mkdir()
    (logs / "old.txt").write_text("旧日志", encoding="utf-8")
    git_cache = root / ".git" / "hooks" / "__pycache__"
    git_cache.mkdir(parents=True)
    protected = git_cache / "keep.pyc"
    protected.write_text("不能删除", encoding="utf-8")

    subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(target_script),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )

    assert not runtime_cache.exists()
    assert not logs.exists()
    assert protected.exists()


def test_gpu_setup_script_requires_python312_cuda128_and_cuda_probe():
    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "setup_gpu_env.ps1"
    assert script.exists()
    content = script.read_text(encoding="utf-8")
    assert "3.12" in content
    assert "https://download.pytorch.org/whl/cu128" in content
    assert "torch.cuda.is_available" in content
    assert "Invoke-CheckedNative" in content


def test_package_boundaries_are_importable_packages():
    root = Path(__file__).resolve().parents[1]
    package_inits = {
        "agent/__init__.py",
        "env/__init__.py",
        "baseline/__init__.py",
        "training/__init__.py",
        "evaluation/__init__.py",
        "tests/__init__.py",
    }
    assert all((root / path).exists() for path in package_inits)


def test_continual_ppo_entrypoints_report_schema_version():
    root = Path(__file__).resolve().parents[1]
    entrypoints = ["calibrate_channel.py", "pretrain.py", "online_train.py", "compare.py"]
    for entrypoint in entrypoints:
        script = root / entrypoint
        assert script.exists()
        result = subprocess.run(
            [sys.executable, str(script), "--version"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert result.stdout.strip() == "RL4EQ continual-ppo schema-v1"


def test_legacy_environment_modules_remain_importable_during_reset():
    import env.comm_env  # noqa: F401
    import env.frame_structure  # noqa: F401


def test_checkpoint_resume_reproduces_next_step(tmp_path):
    uninterrupted = run_tiny_training(seed=7, steps=4)
    partial = run_tiny_training(seed=7, steps=2, save_to=tmp_path / "state.pt")
    resumed = run_tiny_training(seed=999, steps=4, resume=tmp_path / "state.pt")
    assert partial.completed_steps == 2
    assert torch.equal(uninterrupted.model_vector, resumed.model_vector)
    assert uninterrupted.next_batch_hash == resumed.next_batch_hash


def test_corrupt_checkpoint_raises_without_overwriting_good_state(tmp_path):
    good = tmp_path / "good.pt"
    run_tiny_training(seed=3, steps=1, save_to=good)
    before = good.read_bytes()
    bad = tmp_path / "bad.pt"
    bad.write_text("not a checkpoint", encoding="utf-8")
    with pytest.raises(CheckpointError):
        load_checkpoint(bad)
    assert good.read_bytes() == before
    assert load_checkpoint(good)["global_step"] == 1


def test_metrics_do_not_average_away_failed_configs():
    rows = []
    for delay in [20, 30, 40]:
        for snr_db in [10, 15, 20]:
            ber = 0.005
            if delay == 40 and snr_db == 10:
                ber = 0.02
            rows.append(FrameMetric(method="Continual PPO", level="B", delay=delay, snr_db=snr_db, seed=1, frame=0, ber_data=ber))
    rows.append(FrameMetric(method="Continual PPO", level="C", delay=50, snr_db=10, seed=1, frame=0, ber_data=0.2))

    summary = summarize_main_matrix(rows)

    assert len(summary.per_config) == 9
    assert summary.all_below(0.01) is False
    assert summary.generalization["count"] == 1
    assert effective_goodput(
        ber=0.01,
        data_symbols=384,
        ordinary_frames=1000,
        warmup_symbols=40,
        acquisition_symbols=512,
        frame_symbols=512,
    ) == pytest.approx(384 * 1000 * 0.99 / (40 + 512 + 512 * 1000))
    assert spearman_reward_data([1, 2, 3], [0.2, 0.4, 0.9]).correlation == 1.0


def test_pilot_shortlist_keeps_two_to_three_candidates_and_reasons(tmp_path):
    candidates = [
        {"pilot_total": 64, "pilot_layout": "prefix", "max_ber": 0.02, "spearman": 0.7, "effective_goodput": 0.70, "worst_seed": 0.03},
        {"pilot_total": 96, "pilot_layout": "two_block", "max_ber": 0.008, "spearman": 0.8, "effective_goodput": 0.68, "worst_seed": 0.01},
        {"pilot_total": 128, "pilot_layout": "multi_block", "max_ber": 0.007, "spearman": 0.65, "effective_goodput": 0.66, "worst_seed": 0.02},
        {"pilot_total": 160, "pilot_layout": "prefix", "max_ber": 0.006, "spearman": 0.4, "effective_goodput": 0.60, "worst_seed": 0.01},
    ]

    result = select_pilot_shortlist(candidates, tmp_path, keep=2)

    assert len(result["shortlist"]) == 2
    assert all(item["selected"] for item in result["shortlist"])
    assert any("spearman" in item["淘汰原因"] for item in result["all_candidates"] if not item["selected"])
    assert (tmp_path / "shortlist.json").exists()


@pytest.mark.parametrize("method", FORMAL_METHODS)
def test_all_methods_receive_same_label_free_frame(method):
    received = paired_frame(seed=23)
    hidden = received.hide_reward_and_data_labels()
    result_a = run_method(method, hidden)
    result_b = run_method(method, hidden)
    assert result_a.input_hash == result_b.input_hash == received.observable_hash
    assert method not in {"Data" + " Oracle"}


def test_hierarchical_bootstrap_resamples_seed_then_ten_frame_blocks():
    rows = [
        {"seed": seed, "frame": frame, "method": "Continual PPO", "ber_data": 0.01 + 0.001 * seed}
        for seed in range(3)
        for frame in range(30)
    ]
    interval = paired_block_bootstrap(rows, seed=7, repetitions=200, block_length=10)
    assert interval.resampling_order == ("seed", "contiguous_frame_block")
    assert interval.block_length == 10
    assert interval.repetitions == 200
    assert interval.low <= interval.mean <= interval.high
