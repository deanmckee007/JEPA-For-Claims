import numpy as np

from scripts.run_high_cost_tail_probe import (
    balanced_sample_weights,
    ordinal_cost_labels,
    ordinal_ranking_scores,
    score_tail_ranking,
    top_fraction_labels,
    train_tail_threshold,
)


def test_top_fraction_labels_selects_exact_budget():
    labels = top_fraction_labels(np.arange(100), 0.015)
    assert labels.sum() == 2
    assert labels[-2:].tolist() == [1, 1]


def test_perfect_ranking_has_full_precision_and_expected_lift():
    targets = np.linspace(1.0, 9.0, 200)
    metrics = score_tail_ranking(
        targets, targets, fraction=0.015,
        train_threshold=train_tail_threshold(targets, 0.015),
    )
    assert metrics["selection_count"] == 3
    assert metrics["precision_at_budget"] == 1.0
    assert metrics["recall_at_budget"] == 1.0
    assert np.isclose(metrics["lift_at_budget"], 200 / 3)
    assert metrics["cost_capture_at_budget"] > 0.0


def test_balanced_weights_equalize_class_mass():
    labels = np.asarray([0] * 98 + [1] * 2)
    weights = balanced_sample_weights(labels)
    assert np.isclose(weights[labels == 0].sum(), weights[labels == 1].sum())


def test_ordinal_cost_labels_create_nested_exact_bands():
    labels = ordinal_cost_labels(np.arange(1000), [0.005, 0.015, 0.05, 0.20])
    assert np.bincount(labels).tolist() == [800, 150, 35, 10, 5]
    assert labels[-5:].tolist() == [4] * 5


def test_multiclass_weights_equalize_every_class_mass():
    labels = np.asarray([0] * 80 + [1] * 15 + [2] * 5)
    weights = balanced_sample_weights(labels)
    masses = [weights[labels == class_id].sum() for class_id in range(3)]
    assert np.allclose(masses, masses[0])


def test_ordinal_ranking_scores_combine_classes_in_order():
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.3, 0.1]])
    scores = ordinal_ranking_scores(
        probabilities, np.arange(5), tail_class_minimum=3
    )
    assert np.isclose(scores["severity"][0], 2.1)
    assert np.isclose(scores["tail_probability"][0], 0.4)
