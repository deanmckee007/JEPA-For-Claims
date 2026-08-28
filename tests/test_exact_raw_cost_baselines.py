from types import SimpleNamespace

import numpy as np

from scripts.run_exact_raw_cost_baselines import (
    build_exact_history_matrix,
    feature_layout,
    inner_train_val_positions,
    top_frequency_columns,
)


def _dataset(last_cpt="cpt_target"):
    claims = [
        {"cpt": ["cpt_a"], "icd": ["icd_a"], "ttnc": "ttnc_a"},
        {"cpt": ["cpt_b"], "icd": ["icd_b"], "ttnc": "ttnc_b"},
        {"cpt": [last_cpt], "icd": ["icd_target"], "ttnc": "ttnc_target"},
    ]
    return SimpleNamespace(
        cpt_vocab={"<PAD>": 0, "<UNK>": 1, "cpt_a": 2, "cpt_b": 3, "cpt_target": 4, "changed": 5},
        icd_vocab={"<PAD>": 0, "<UNK>": 1, "icd_a": 2, "icd_b": 3, "icd_target": 4},
        ttnc_vocab={"<PAD>": 0, "<UNK>": 1, "ttnc_a": 2, "ttnc_b": 3, "ttnc_target": 4},
        processed_data=[claims],
        targets=[8.0],
        max_claims_len=50,
        evaluation_claim_inclusion_policy=None,
    )


def test_exact_history_excludes_held_out_target_claim():
    first, targets, layout = build_exact_history_matrix([0], _dataset())
    second, _, _ = build_exact_history_matrix([0], _dataset("changed"))
    assert np.array_equal(first.toarray(), second.toarray())
    assert targets.tolist() == [8.0]
    assert first.shape == (1, layout["dimension"])
    assert first[0, layout["last_cpt"] + 3] == 1.0


def test_feature_layout_has_two_code_blocks_and_four_statistics():
    dataset = _dataset()
    layout = feature_layout(dataset)
    expected = 2 * (
        len(dataset.cpt_vocab) + len(dataset.icd_vocab) + len(dataset.ttnc_vocab)
    ) + 4
    assert layout["dimension"] == expected


def test_inner_split_is_disjoint_and_complete():
    train, val = inner_train_val_positions(np.arange(100), seed=42)
    assert not set(train).intersection(val)
    assert sorted(np.concatenate([train, val]).tolist()) == list(range(100))


def test_top_frequency_columns_keeps_statistics_tail():
    matrix, _, layout = build_exact_history_matrix([0], _dataset())
    columns = top_frequency_columns(matrix, 8)
    assert set(range(layout["dimension"] - 4, layout["dimension"])).issubset(columns)
