from scripts.create_thesis_figures import (
    Run,
    _category,
    _collapse_recoveries,
    _momentum_spikes,
)


def _run(test_accuracies, *, delayed=6, delay=5, epochs=20):
    return Run(
        path=None,
        config={
            "experiment": {"epochs": epochs},
            "distributed": {
                "staleness_mode": "mixed_fixed",
                "nodes": 6,
                "zero_delay_nodes": 6 - delayed,
            },
        },
        metrics=[
            {
                "step": step,
                "test_accuracy": accuracy,
                "train_accuracy": accuracy,
            }
            for step, accuracy in enumerate(test_accuracies)
        ],
        spectra=[],
        run_id="test",
        seed=1,
        nodes=6,
        delayed_nodes=delayed,
        delay=delay,
    )


def test_phase_category_requires_logged_sustained_window_or_collapse():
    sustained = _run([0.5] + [0.995] * 1001, epochs=1001)
    collapsed = _run([0.5, 0.995, 0.995, 0.98] + [0.97] * 20, epochs=23)
    never = _run([0.5] * 21, epochs=20)

    assert _category(sustained) == "sustained"
    assert _category(collapsed) == "collapse"
    assert _category(never) == "never"


def test_collapse_recovery_detection_uses_accuracy_thresholds():
    run = _run([0.5, 0.995, 0.85, 0.88, 0.995, 0.995])
    assert _collapse_recoveries(run) == [(2, 3, 4)]


def test_momentum_peak_rule_returns_intervals_from_detected_events():
    points = [(step, value) for step, value in enumerate(
        [0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    )]
    peaks, threshold = _momentum_spikes(points)
    assert peaks == [(9, 3.0)]
    assert threshold > 0.0
