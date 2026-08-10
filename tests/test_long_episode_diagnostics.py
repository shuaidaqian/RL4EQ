import json

import torch


def test_long_episode_diagnostic_reports_tail_and_cir_modes(tmp_path):
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from evaluation.long_episode_diagnostics import run_long_episode_diagnostic

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    result = run_long_episode_diagnostic(
        config_path="configs/continual_ppo.json",
        pretrained=pretrained_dir / "model_best.pt",
        output_dir=tmp_path / "diagnostic",
        delay=20,
        snr_db=10.0,
        frames=2,
        seeds=[0],
        tail_modes=["soft", "oracle"],
        cir_modes=["fixed", "adapt_pilot", "oracle"],
        rhos=[0.99],
        pilot_total=64,
        pilot_layout="two_block",
        device="cpu",
    )

    assert result["schema_version"] == "long-episode-diagnostic-v1"
    assert result["diagnostic_only"] is True
    assert len(result["rows"]) == 12
    assert {row["tail_mode"] for row in result["rows"]} == {"soft", "oracle"}
    assert {row["cir_mode"] for row in result["rows"]} == {"fixed", "adapt_pilot", "oracle"}
    assert all(row["method"] == "Offline NN diagnostic" for row in result["rows"])
    assert all("ber_data" in row for row in result["rows"])
    assert all("frame_bin" in row for row in result["rows"])
    adapt_rows = [row for row in result["rows"] if row["cir_mode"] == "adapt_pilot"]
    assert adapt_rows
    assert all(row["cir_estimation_uses_adapt_pilot"] is True for row in adapt_rows)
    assert (tmp_path / "diagnostic" / "frame_metrics.jsonl").exists()
    assert (tmp_path / "diagnostic" / "summary.json").exists()


def test_long_episode_diagnostic_uses_configurable_dd_cir_alpha(tmp_path):
    from agent.unfolded_equalizer import UnfoldedConfig, UnfoldedEqualizer
    from evaluation.long_episode_diagnostics import run_long_episode_diagnostic

    pretrained_dir = tmp_path / "pretrained"
    pretrained_dir.mkdir()
    model = UnfoldedEqualizer(UnfoldedConfig(frame_len=512, max_delay=40, iterations=1, d_model=24, num_heads=4))
    (pretrained_dir / "model_config.json").write_text(json.dumps(model.config.to_dict(), ensure_ascii=False), encoding="utf-8")
    torch.save({"schema_version": "unfolded-eq-v1", "state_dict": model.state_dict()}, pretrained_dir / "model_best.pt")

    result = run_long_episode_diagnostic(
        config_path="configs/continual_ppo.json",
        pretrained=pretrained_dir / "model_best.pt",
        output_dir=tmp_path / "diagnostic_alpha",
        delay=20,
        snr_db=10.0,
        frames=2,
        seeds=[0],
        tail_modes=["soft"],
        cir_modes=["decision_directed"],
        rhos=[0.99],
        pilot_total=64,
        pilot_layout="prefix",
        cir_alpha=0.4,
        cir_confidence_threshold=0.6,
        device="cpu",
    )

    rows = result["rows"]
    assert rows
    assert all(row["cir_mode"] == "decision_directed" for row in rows)
    assert all(row["cir_update_alpha"] == 0.4 for row in rows)
    assert all(row["cir_confidence_threshold"] == 0.6 for row in rows)
    assert all(row["cir_update_uses_data_labels"] is False for row in rows)
