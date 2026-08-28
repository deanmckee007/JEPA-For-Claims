import numpy as np
import torch

from scripts.run_frozen_generation_probe import (
    AddRemoveClaimDecoder,
    FrozenClaimDecoder,
    add_remove_to_membership_prediction,
    apply_copy_bias,
    extract_immediate_next_claim_targets,
    extract_previous_and_target_claims,
    hashed_raw_history_features,
    initialize_decoder_from_priors,
    modality_metrics,
    token_ids_to_multi_hot,
)


def test_extract_immediate_next_claim_targets_respects_left_padding_and_horizon():
    cpt = torch.tensor(
        [
            [[0, 0], [2, 0], [3, 4], [5, 0]],
            [[0, 0], [0, 0], [7, 0], [8, 9]],
        ]
    )
    icd = cpt + (cpt != 0).long() * 10
    ttnc = torch.tensor([[0, 1, 2, 3], [0, 0, 4, 5]])

    next_cpt, next_icd, next_ttnc = extract_immediate_next_claim_targets(
        cpt, icd, ttnc, future_claim_k=1
    )

    assert next_cpt.tolist() == [[3, 4], [8, 9]]
    assert next_icd.tolist() == [[13, 14], [18, 19]]
    assert next_ttnc.tolist() == [2, 5]


def test_extract_target_matches_model_behavior_for_single_valid_claim():
    cpt = torch.tensor([[[0, 0], [0, 0], [7, 0]]])
    icd = torch.tensor([[[0, 0], [0, 0], [9, 0]]])
    ttnc = torch.tensor([[0, 0, 4]])
    next_cpt, next_icd, next_ttnc = extract_immediate_next_claim_targets(
        cpt, icd, ttnc
    )
    assert next_cpt.tolist() == [[7, 0]]
    assert next_icd.tolist() == [[9, 0]]
    assert next_ttnc.tolist() == [4]


def test_previous_claim_baseline_uses_claim_immediately_before_target():
    cpt = torch.tensor([[[0, 0], [2, 0], [3, 0], [4, 0]]])
    icd = cpt + (cpt != 0).long() * 10
    ttnc = torch.tensor([[0, 1, 2, 3]])
    result = extract_previous_and_target_claims(
        cpt, icd, ttnc, future_claim_k=1
    )
    assert result["eligible"].tolist() == [True]
    assert result["previous_cpt_ids"].tolist() == [[2, 0]]
    assert result["cpt_ids"].tolist() == [[3, 0]]
    assert result["previous_ttnc"].tolist() == [1]
    assert result["ttnc"].tolist() == [2]


def test_token_ids_to_multi_hot_ignores_padding_and_duplicates():
    result = token_ids_to_multi_hot(np.asarray([[0, 2, 2], [1, 3, 0]]), 5)
    assert result.tolist() == [
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
    ]


def test_raw_history_hash_excludes_held_out_target():
    cpt_a = torch.tensor([[[0, 0], [2, 0], [3, 0], [4, 0]]])
    cpt_b = cpt_a.clone()
    cpt_b[0, 3] = torch.tensor([99, 0])
    icd = torch.tensor([[[0, 0], [12, 0], [13, 0], [14, 0]]])
    ttnc = torch.tensor([[0, 1, 2, 3]])
    feature_a = hashed_raw_history_features(cpt_a, icd, ttnc)
    feature_b = hashed_raw_history_features(cpt_b, icd, ttnc)
    assert np.array_equal(feature_a, feature_b)

    cpt_b[0, 2] = torch.tensor([98, 0])
    feature_c = hashed_raw_history_features(cpt_b, icd, ttnc)
    assert not np.array_equal(feature_a, feature_c)


def test_decoder_shapes_match_all_claim_modalities():
    decoder = FrozenClaimDecoder(8, 16, 11, 13, 5)
    outputs = decoder(torch.randn(4, 8))
    assert outputs["cpt_logits"].shape == (4, 11)
    assert outputs["icd_logits"].shape == (4, 13)
    assert outputs["ttnc_logits"].shape == (4, 5)
    assert outputs["cardinality_fraction"].shape == (4, 2)


def test_add_remove_decoder_shapes_match_change_events():
    decoder = AddRemoveClaimDecoder(8, 16, 11, 13, 5)
    outputs = decoder(torch.randn(4, 8))
    assert outputs["cpt_add_logits"].shape == (4, 11)
    assert outputs["cpt_remove_logits"].shape == (4, 11)
    assert outputs["icd_add_logits"].shape == (4, 13)
    assert outputs["icd_remove_logits"].shape == (4, 13)
    assert outputs["ttnc_logits"].shape == (4, 5)


def test_add_remove_prediction_preserves_kept_codes_and_adds_new_codes():
    change = {
        "cpt_add_logits": torch.tensor([[0.0, -10.0, 10.0, -10.0]]),
        "cpt_remove_logits": torch.tensor([[0.0, -10.0, 0.0, 0.0]]),
        "icd_add_logits": torch.tensor([[0.0, -10.0, -10.0, 10.0]]),
        "icd_remove_logits": torch.tensor([[0.0, -10.0, 0.0, 0.0]]),
        "ttnc_logits": torch.zeros(1, 5),
    }
    prediction = add_remove_to_membership_prediction(
        change,
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([2]),
        ttnc_copy_boost=2.0,
        max_cpt_tokens=4,
        max_icd_tokens=4,
    )
    cpt_probability = torch.sigmoid(prediction["cpt_logits"])
    icd_probability = torch.sigmoid(prediction["icd_logits"])
    assert cpt_probability[0, 1] > 0.99
    assert cpt_probability[0, 2] > 0.99
    assert icd_probability[0, 1] > 0.99
    assert icd_probability[0, 3] > 0.99
    assert prediction["ttnc_logits"].argmax(dim=1).item() == 2


def test_decoder_prior_initialization_is_feature_independent():
    decoder = FrozenClaimDecoder(4, 8, 5, 4, 3, dropout=0.0)
    cpt = np.asarray([[0, 1, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=np.float32)
    icd = np.asarray([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
    ttnc = np.asarray([1, 2])
    counts = np.asarray([[0.5, 0.25], [0.25, 0.25]], dtype=np.float32)
    initialize_decoder_from_priors(decoder, cpt, icd, ttnc, counts)
    decoder.eval()
    first = decoder(torch.randn(2, 4))
    second = decoder(torch.randn(2, 4))
    assert torch.allclose(first["cpt_logits"], second["cpt_logits"])
    assert torch.allclose(first["icd_logits"], second["icd_logits"])
    assert torch.allclose(first["ttnc_logits"], second["ttnc_logits"])


def test_copy_bias_only_boosts_previous_codes_and_ttnc():
    prediction = {
        "cpt_logits": torch.zeros(1, 5),
        "icd_logits": torch.zeros(1, 4),
        "ttnc_logits": torch.zeros(1, 3),
        "cardinality_fraction": torch.zeros(1, 2),
    }
    result = apply_copy_bias(
        prediction,
        torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([2]),
        code_boost=4.0,
        ttnc_boost=2.0,
    )
    assert result["cpt_logits"].tolist() == [[0.0, 0.0, 4.0, 0.0, 0.0]]
    assert result["icd_logits"].tolist() == [[0.0, 4.0, 0.0, 0.0]]
    assert result["ttnc_logits"].tolist() == [[0.0, 0.0, 2.0]]


def test_modality_metrics_are_perfect_for_perfect_rankings_and_counts():
    targets = np.asarray(
        [
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    probabilities = targets * 0.98 + (1 - targets) * 0.02
    metrics = modality_metrics(
        targets,
        probabilities,
        predicted_counts=np.asarray([2, 1]),
        train_support=np.asarray([0, 20, 20, 20]),
        min_class_support=10,
    )
    assert metrics["micro_f1"] == 1.0
    assert metrics["mean_set_jaccard"] == 1.0
    assert metrics["exact_set_match"] == 1.0
    assert metrics["cardinality_mae"] == 0.0
