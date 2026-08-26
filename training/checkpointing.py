# -*- coding: utf-8 -*-
"""可恢复实验 checkpoint 与确定性 smoke 训练。"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class CheckpointError(RuntimeError):
    """checkpoint 无法读取或 schema 不兼容。"""


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    try:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise CheckpointError(f"无法读取 checkpoint：{path}") from exc
    if not isinstance(state, dict) or state.get("schema_version") != "rl4eq-checkpoint-v1":
        raise CheckpointError(f"checkpoint schema 不兼容：{path}")
    return state


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass(frozen=True)
class TinyTrainingResult:
    completed_steps: int
    model_vector: torch.Tensor
    next_batch_hash: str


def run_tiny_training(
    seed: int,
    steps: int,
    save_to: str | Path | None = None,
    resume: str | Path | None = None,
) -> TinyTrainingResult:
    if resume is None:
        _seed_everything(seed)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        start_step = 0
    else:
        checkpoint = load_checkpoint(resume)
        model = nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        restore_rng_state(checkpoint["rng"])
        start_step = int(checkpoint["global_step"])

    for _ in range(start_step, steps):
        batch = torch.randn(8, 4)
        target = torch.randn(8, 2)
        loss = torch.mean((model(batch) - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if save_to is not None:
        save_checkpoint(
            save_to,
            {
                "schema_version": "rl4eq-checkpoint-v1",
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": None,
                "grad_scaler": None,
                "rng": capture_rng_state(),
                "stage": "tiny",
                "global_step": steps,
                "config_hash": _hash_json({"seed": seed, "steps": steps}),
            },
        )

    next_batch_hash = _tensor_hash(torch.randn(8, 4))
    return TinyTrainingResult(
        completed_steps=steps,
        model_vector=torch.cat([parameter.detach().flatten() for parameter in model.parameters()]),
        next_batch_hash=next_batch_hash,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hash_json(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()
