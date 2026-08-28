import numpy as np

from scripts.run_tail_candidate_showdown import (
    aggregate_budget_runs,
    blend_candidate_scores,
    percentile_ranks,
)


def test_percentile_ranks_average_ties():
    ranks = percentile_ranks([10.0, 20.0, 20.0, 40.0])
    assert np.allclose(ranks, [0.0, 0.5, 0.5, 1.0])


def test_blend_candidate_scores_respects_endpoints_and_weight():
    raw = np.asarray([1.0, 2.0, 3.0])
    hybrid = np.asarray([3.0, 2.0, 1.0])
    assert np.allclose(blend_candidate_scores(raw, hybrid, 0.25), [0.75, 0.5, 0.25])
    assert np.allclose(blend_candidate_scores(raw, hybrid, 0.75), [0.25, 0.5, 0.75])


def test_aggregate_budget_runs_keeps_fractions_separate():
    metrics = {
        name: 0.5 for name in (
            "precision_at_budget", "recall_at_budget", "lift_at_budget",
            "average_precision", "binary_ndcg_at_budget", "cost_capture_at_budget",
            "cost_lift_at_budget", "threshold_precision_at_budget",
            "threshold_recall_at_budget",
        )
    }
    runs = [
        {"condition": "a", "evaluation_fraction": fraction, "metrics": metrics}
        for fraction in (0.01, 0.02)
    ]
    aggregate = aggregate_budget_runs(runs)
    assert sorted(aggregate["a"]) == ["0.01", "0.02"]
