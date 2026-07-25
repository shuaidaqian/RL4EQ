from pathlib import Path
import subprocess
import sys


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
