import numpy as np

from jepa_utils.claim_set_calibration import (
    shortlist_logits, fit_set_calibration, decode_calibrated_sets,
    score_decoded_sets, candidate_coverage,
)
from scripts.run_retrieval_replication import grouped_inner_split, filter_prediction
from scripts.run_retrieval_replication import evaluate_prediction
from types import SimpleNamespace


def test_inner_split_is_grouped_disjoint_and_reproducible():
    ids = np.asarray(["a", "b", "a", "c", "d", "e"])
    fit, calibration = grouped_inner_split(ids, 0.4, 12)
    assert not set(ids[fit]) & set(ids[calibration])
    assert sorted(np.r_[fit, calibration].tolist()) == list(range(len(ids)))
    again = grouped_inner_split(ids, 0.4, 12)
    assert np.array_equal(calibration, again[1])


def test_new_candidate_recall_excludes_persistence_and_reserved_ids():
    result = candidate_coverage([[1, 2, 4]], [[1, 2, 3, 4]], [[2, 0, 0]])
    assert result["persistent"]["recall"] == 1.0
    assert result["new"] == {"covered": 1, "total": 2, "recall": 0.5}
    assert result["all_known"]["total"] == 3


def test_additions_objective_exposes_the_total_f1_tradeoff():
    logits = np.asarray([[-30., -30, 4., 0.5]] * 4)
    previous = np.asarray([[2]] * 4)
    target = np.asarray([[2, 3], [2, 0], [2, 0], [2, 0]])
    shortlist = shortlist_logits(logits, previous, 2)
    micro = fit_set_calibration(shortlist, [2] * 4, target, previous_ids=previous)
    balanced = fit_set_calibration(shortlist, [2] * 4, target,
        objective="balanced_new_f1", previous_ids=previous)
    micro_sets = decode_calibrated_sets(shortlist, [2] * 4, micro)
    balanced_sets = decode_calibrated_sets(shortlist, [2] * 4, balanced)
    micro_metrics = score_decoded_sets(micro_sets, target, previous)
    balanced_metrics = score_decoded_sets(balanced_sets, target, previous)
    assert balanced_metrics['new_code_recall'] > micro_metrics['new_code_recall']
    assert balanced_metrics['micro_f1'] < micro_metrics['micro_f1']


def test_calibration_removes_false_additions_without_losing_copied_token():
    logits = np.asarray([[-30, -30, 1.5, 1.0], [-30, -30, 1.5, 1.0]])
    previous = np.asarray([[2], [2]])
    targets = np.asarray([[2], [2]])
    shortlist = shortlist_logits(logits, previous, 2)
    calibration = fit_set_calibration(shortlist, [2, 2], targets)
    prediction = decode_calibrated_sets(shortlist, [2, 2], calibration)
    assert prediction == [[2], [2]]
    assert score_decoded_sets(prediction, targets, previous)["micro_f1"] == 1.0


def test_decoding_never_emits_padding_or_non_candidates_when_count_is_large():
    logits = np.asarray([[99., -30, 2., -30, -30]])
    shortlist = shortlist_logits(logits, np.asarray([[0]]), 4)
    settings = {"copy_logit_threshold": -2, "add_logit_threshold": -2, "count_scale": 2}
    assert decode_calibrated_sets(shortlist, [100], settings) == [[2]]


def test_shortlist_matches_exhaustive_bounded_decoding():
    rng = np.random.default_rng(14)
    logits = rng.normal(size=(5, 30))
    previous = np.asarray([[2, 6, 12]] * 5)
    shortlist = shortlist_logits(logits, previous, 4)
    settings = {"copy_logit_threshold": 0.5, "add_logit_threshold": -0.5, "count_scale": None}
    actual = decode_calibrated_sets(shortlist, [1] * 5, settings)
    expected = []
    for row in logits:
        margins = row - np.where(np.isin(np.arange(30), previous[0]), 0.5, -0.5)
        eligible = [i for i in range(1, 30) if margins[i] > 0]
        expected.append(sorted(eligible, key=lambda i: (-margins[i], i))[:4])
    assert actual == expected


def test_filter_control_does_not_mutate_flat_prediction():
    prediction = {"cpt_logits": np.ones((1, 5)), "icd_logits": np.ones((1, 5))}
    filtered = filter_prediction(prediction, [np.asarray([[2, 0]]), np.asarray([[3, 0]])])
    assert prediction["cpt_logits"][0, 4] == 1
    assert filtered["cpt_logits"][0, 4] == -30
    assert filtered["cpt_logits"][0, 2] == 1


def test_outer_labels_cannot_change_calibration_settings():
    config = SimpleNamespace(cpt_vocab_size=5, icd_vocab_size=5, max_cpt_tokens=2, max_icd_tokens=2)
    data = {"cpt_ids": np.asarray([[2], [2], [3]]), "icd_ids": np.asarray([[2], [2], [3]]),
        "previous_cpt_ids": np.asarray([[2], [2], [2]]), "previous_icd_ids": np.asarray([[2], [2], [2]]),
        "ttnc": np.asarray([1, 1, 1])}
    prediction = {"cpt_logits": np.asarray([[-30., -3, 1.5, 1.0, -3]] * 3),
        "icd_logits": np.asarray([[-30., -3, 1.5, 1.0, -3]] * 3),
        "ttnc_logits": np.asarray([[0., 1.]] * 3), "cardinality_fraction": np.ones((3, 2))}
    result = evaluate_prediction(prediction, data, 2, config, [np.ones(5)] * 2)
    data["cpt_ids"][2] = 4
    data["icd_ids"][2] = 4
    changed = evaluate_prediction(prediction, data, 2, config, [np.ones(5)] * 2)
    assert result["calibration"] == changed["calibration"]
