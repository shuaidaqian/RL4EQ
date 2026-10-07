import json

from scripts.summarize_pilot_objective_replay import main


def test_summarize_pilot_objective_replay_handles_constant_values(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    rows = [
        {"objective": "bce", "action_name": "identity", "seed": 0, "window_index": 0},
        {"objective": "bce", "action_name": "channel_residual", "seed": 0, "window_index": 0,
         "reward_loss_improvement": 0.0, "data_ber_improvement": 0.0},
    ]
    (log_dir / "window_peft_rows.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    output = log_dir / "summary.json"
    monkeypatch.setattr("sys.argv", ["summarize", "--log-dir", str(log_dir), "--output", str(output)])
    main()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["objective_summary"][0]["mean_spearman"] is None

