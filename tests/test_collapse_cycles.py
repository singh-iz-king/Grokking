from __future__ import annotations

from dataclasses import replace

import pandas as pd

from scripts.plot_collapse_cycles import (
    METRICS,
    Run,
    _configuration_summary,
    aligned_rows,
    detect_events,
    make_event_records,
    run_analysis,
)


def _synthetic_metrics() -> pd.DataFrame:
    steps = [3499, 3500, 3501, 3510, 3540, 3600, 3610, 3660, 3700]
    accuracy = [0.995, 0.995, 0.98, 0.84, 0.995, 0.995, 0.87, 0.96, 0.95]
    rows = []
    for step, test_accuracy in zip(steps, accuracy):
        rows.append({
            "step": step,
            "test_accuracy": test_accuracy,
            "delayed_gradient_relative_discrepancy_mean": 0.6 if step < 3501 else 0.8,
            "delayed_gradient_cosine_mean": 0.85,
            "aggregated_momentum_l2_norm": 0.5,
        })
    return pd.DataFrame(rows)


def _run(frame: pd.DataFrame) -> Run:
    config = {
        "experiment": {"name": "synthetic", "seed": 7, "modulus": 113},
        "optimizer": {"name": "AdamW", "learning_rate": 0.001, "weight_decay": 1.0},
        "distributed": {
            "enabled": True,
            "nodes": 6,
            "zero_delay_nodes": 0,
            "tau": 5,
        },
        "model": {"n_layers": 1, "d_model": 128},
    }
    return Run(
        path=__import__("pathlib").Path("/synthetic/run"),
        run_id="synthetic_run",
        seed="7",
        config=config,
        settings_id="setting",
        settings_label="synthetic setting",
        metrics=frame,
        metrics_source="metrics.csv",
    )


def test_detects_recovered_and_incomplete_major_events_with_real_steps() -> None:
    events = detect_events(_synthetic_metrics(), min_step=3500)
    assert [event["onset_step"] for event in events] == [3501, 3610]
    assert [event["trough_step"] for event in events] == [3510, 3610]
    assert events[0]["recovery_step"] == 3540
    assert events[1]["recovery_step"] is None


def test_min_step_accuracy_scale_and_non_major_dip() -> None:
    frame = pd.DataFrame({
        "step": [100, 200, 300, 400, 500, 600],
        "test_accuracy": [99.5, 98.0, 99.5, 99.5, 85.0, 99.5],
    })
    events = detect_events(frame, min_step=350)
    assert len(events) == 1
    assert events[0]["onset_step"] == 500
    assert events[0]["trough_accuracy"] == 0.85
    assert events[0]["recovery_step"] == 600


def test_event_pair_fields_and_alignment_do_not_interpolate_missing_steps() -> None:
    run = _run(_synthetic_metrics())
    events = make_event_records(
        [run], onset_threshold=0.99, severe_threshold=0.90, min_step=3500
    )
    assert events[0].next_onset_step == 3610
    assert events[0].next_trough_step == 3610
    assert events[0].next_recovery_step is None
    assert events[0].next_recovery_duration is None
    aligned = aligned_rows(
        [run],
        events,
        pre_window=2,
        post_window=10,
        adaptive_window=False,
    )
    missing_step = aligned[
        (aligned["event_id"] == events[0].event_id)
        & (aligned["step_relative"] == 1)
    ].iloc[0]
    assert missing_step["step_absolute"] == 3502
    assert pd.isna(missing_step["test_accuracy"])
    assert list(METRICS) == [
        "delayed_gradient_relative_discrepancy_mean",
        "delayed_gradient_cosine_mean",
        "aggregated_momentum_l2_norm",
        "test_accuracy",
    ]


def test_next_event_recovery_statistics_use_consecutive_events() -> None:
    frame = pd.concat([
        _synthetic_metrics(),
        pd.DataFrame([{
            "step": 3710,
            "test_accuracy": 0.995,
            "delayed_gradient_relative_discrepancy_mean": 0.3,
            "delayed_gradient_cosine_mean": 0.98,
            "aggregated_momentum_l2_norm": 0.5,
        }]),
    ], ignore_index=True)
    events = make_event_records(
        [_run(frame)],
        onset_threshold=0.99,
        severe_threshold=0.90,
        min_step=3500,
    )
    assert events[0].next_onset_step == 3610
    assert events[0].next_recovery_step == 3710
    assert events[0].next_recovery_duration == 100


def test_configuration_statistics_balance_events_by_seed() -> None:
    base_run = _run(_synthetic_metrics())
    base_event = make_event_records(
        [base_run],
        onset_threshold=0.99,
        severe_threshold=0.90,
        min_step=3500,
    )[0]
    seed_one = replace(base_run, run_id="run_seed1", seed="1")
    seed_two = replace(base_run, run_id="run_seed2", seed="2")
    events = [
        replace(base_event, event_id=f"seed1_{delay}", run_id="run_seed1",
                seed="1", trough_delay_steps=delay)
        for delay in (10, 20, 30)
    ] + [
        replace(base_event, event_id="seed2_100", run_id="run_seed2",
                seed="2", trough_delay_steps=100)
    ]

    summary = _configuration_summary(events, [seed_one, seed_two]).iloc[0]
    assert summary["trough_delay_steps_median"] == 60
    assert summary["trough_delay_steps_n_seeds"] == 2
    assert summary["trough_delay_steps_n_events"] == 4


def test_run_analysis_exports_events_trajectories_and_figures(tmp_path) -> None:
    results_dir = tmp_path / "results"
    run_dir = results_dir / "synthetic_run"
    run_dir.mkdir(parents=True)
    config = {
        "experiment": {"name": "synthetic", "seed": 7, "modulus": 113},
        "optimizer": {
            "name": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 1.0,
            "beta1": 0.9,
            "beta2": 0.98,
            "epsilon": 1e-8,
        },
        "distributed": {
            "enabled": True,
            "nodes": 6,
            "zero_delay_nodes": 0,
            "tau": 5,
            "staleness_mode": "mixed_fixed",
        },
        "model": {"n_layers": 1, "d_model": 128},
    }
    import yaml

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    frame = _synthetic_metrics()
    frame.to_csv(run_dir / "metrics.csv", index=False)
    output_dir = tmp_path / "analysis"
    runs, events, aligned = run_analysis(
        results_dir,
        output_dir,
        min_step=3500,
        pre_window=10,
        post_window=80,
    )
    assert len(runs) == 1
    assert len(events) == 2
    assert not aligned.empty
    assert (output_dir / "individual_runs" / "synthetic_run.png").is_file()
    assert len(list((output_dir / "by_configuration").glob("*.png"))) == 1
    assert (output_dir / "comparisons" / "all_configurations_01.png").is_file()
    assert (output_dir / "data" / "collapse_events.csv").is_file()
    assert (output_dir / "data" / "configuration_summary.csv").is_file()
    assert (output_dir / "data" / "aligned_trajectories.csv").is_file()
