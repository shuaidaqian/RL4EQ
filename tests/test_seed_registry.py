# -*- coding: utf-8 -*-
"""实验种子登记防止历史、开发和确认之间的重复使用。"""

import json

import pytest

from evaluation.seed_registry import reserve_seeds


def test_seed_registry_rejects_observed_reserved_and_duplicate_seeds(tmp_path):
    results = tmp_path / "docs/results/old"
    results.mkdir(parents=True)
    (results / "old.json").write_text(json.dumps({"rows": [{"seed": 210001}]}))
    for seed in [100001, 170031, 210001]:
        with pytest.raises(ValueError, match="已使用"):
            reserve_seeds(tmp_path, experiment="bad", seeds=[seed], purpose="development", details={})
    reserve_seeds(tmp_path, experiment="first", seeds=[210002], purpose="development", details={})
    with pytest.raises(ValueError, match="已使用"):
        reserve_seeds(tmp_path, experiment="second", seeds=[210002], purpose="confirmation", details={})
    with pytest.raises(ValueError, match="实验名"):
        reserve_seeds(tmp_path, experiment="first", seeds=[210003], purpose="development", details={})
