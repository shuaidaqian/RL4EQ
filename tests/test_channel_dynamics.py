import torch

from evaluation.channel_dynamics import estimate_ar1_statistics


def test_estimate_ar1_recovers_shared_rho_and_tap_power():
    generator = torch.Generator().manual_seed(83)
    rho = 0.91
    target_power = torch.tensor([0.3, 1.2, 0.0], dtype=torch.float64)
    scale = target_power.sqrt()
    trajectories = []
    for _ in range(24):
        state = torch.complex(
            torch.randn(3000, 3, generator=generator, dtype=torch.float64),
            torch.randn(3000, 3, generator=generator, dtype=torch.float64),
        ) / 2**0.5
        state[:, 2] = 0
        values = [state[0] * scale]
        for innovation in state[1:]:
            values.append(rho * values[-1] + (1 - rho**2) ** 0.5 * scale * innovation)
        trajectories.append(torch.stack(values))

    estimate = estimate_ar1_statistics(trajectories)
    assert abs(estimate["rho"] - rho) < 0.003
    assert torch.allclose(estimate["stationary_power"], target_power, atol=0.02, rtol=0.04)
    assert estimate["transition_count"] == 24 * 2999


def test_estimate_ar1_rejects_nontrajectory_input():
    try:
        estimate_ar1_statistics([torch.ones(1, 4, dtype=torch.complex64)])
    except ValueError as error:
        assert "至少两帧" in str(error)
    else:
        raise AssertionError("一帧观测不能估计 AR(1) 系数。")
