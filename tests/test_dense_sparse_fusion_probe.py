import numpy as np
import torch

from scripts.run_dense_sparse_fusion_probe import (
    MatchedFusionTailNetwork,
    standardize_from_train,
)


def test_standardize_from_train_uses_only_training_statistics():
    train = np.asarray([[1.0, 10.0], [3.0, 14.0]], dtype=np.float32)
    val = np.asarray([[5.0, 18.0]], dtype=np.float32)
    standardized_train, standardized_val = standardize_from_train(train, val)
    assert np.allclose(standardized_train.mean(axis=0), 0.0)
    assert np.allclose(standardized_train.std(axis=0), 1.0)
    assert np.allclose(standardized_val, [[3.0, 3.0]])


def test_matched_fusion_modes_have_same_parameter_count_and_shapes():
    raw = torch.randn(4, 6)
    dense = torch.randn(4, 8)
    sparse = torch.randn(4, 8)
    concat = MatchedFusionTailNetwork(6, 8, mode="concat")
    gate = MatchedFusionTailNetwork(6, 8, mode="gate")
    assert sum(p.numel() for p in concat.parameters()) == sum(
        p.numel() for p in gate.parameters()
    )
    concat_logits, concat_gate = concat(raw, dense, sparse)
    gate_logits, gate_values = gate(raw, dense, sparse)
    assert concat_logits.shape == gate_logits.shape == (4,)
    assert concat_gate is None
    assert gate_values.shape == (4, 8)
    assert torch.all((gate_values >= 0.0) & (gate_values <= 1.0))
