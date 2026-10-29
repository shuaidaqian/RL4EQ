"""验证连续 Pilot PEFT 的参数累计与单步限幅。"""

import torch

from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
from scripts.replay_pilot_persistent_peft import (
    _clip_candidate_delta,
    _parameter_delta_norm,
    _group_parameters,
)


def test_persistent_adapter_delta_is_clipped_without_changing_offline_parameters():
    torch.manual_seed(32)
    model = UnfoldedEqualizer(UnfoldedConfig(
        frame_len=128,
        max_delay=4,
        iterations=1,
        d_model=16,
        num_heads=4,
        enable_phase_correction_branch=True,
        phase_correction_initial_scale=1.0,
    ))
    model.attach_online_phase_trend_adapter()
    reference = UnfoldedEqualizer(model.config)
    reference.attach_online_phase_trend_adapter()
    reference.load_state_dict(model.state_dict(), strict=True)
    before = {
        name: parameter.detach().clone()
        for name, parameter in _group_parameters(model, "phase_trend").items()
    }
    with torch.no_grad():
        for parameter in _group_parameters(model, "phase_trend").values():
            parameter.add_(0.25)

    clipped_norm = _clip_candidate_delta(model, before, "phase_trend", 0.02)

    assert clipped_norm == 0.02
    assert _parameter_delta_norm(model, reference, "phase_trend") <= 0.020001
    for name, parameter in reference.named_parameters():
        if getattr(parameter, "_peft_group", None) != "phase_trend":
            assert torch.equal(parameter, dict(model.named_parameters())[name])
