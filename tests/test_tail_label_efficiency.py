import numpy as np

from scripts.run_tail_label_efficiency import stratified_binary_positions


def test_stratified_binary_positions_preserves_rare_class():
    labels = np.asarray([0] * 98 + [1] * 2)
    positions = stratified_binary_positions(labels, 0.1, seed=42)
    selected = labels[positions]
    assert len(positions) == 11
    assert selected.sum() == 1


def test_stratified_binary_positions_returns_all_at_full_fraction():
    labels = np.asarray([0, 1, 0, 1])
    assert np.array_equal(
        stratified_binary_positions(labels, 1.0, seed=42), np.arange(4)
    )
