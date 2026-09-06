import numpy as np
from jepa_utils.neighbor_cost import crossfit_summaries


def test_heldout_group_costs_do_not_change_own_features():
    x = np.random.default_rng(4).normal(size=(12, 5))
    groups = np.repeat(np.arange(6), 2)
    costs = np.arange(12, dtype=float) / 3
    codes = [np.ones((12, 2), dtype=int) * 2] * 2
    original, _, folds = crossfit_summaries([x], [x[:1]], costs, codes, groups, folds=3)
    altered = costs.copy()
    heldout = folds == folds[0]
    altered[heldout] += 5
    changed, _, _ = crossfit_summaries([x], [x[:1]], altered, codes, groups, folds=3)
    np.testing.assert_array_equal(original[heldout], changed[heldout])
    assert not np.allclose(original[~heldout], changed[~heldout])
    for group in np.unique(groups):
        assert len(set(folds[groups == group])) == 1
