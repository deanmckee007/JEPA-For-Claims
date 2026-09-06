import numpy as np
from scripts.run_neighbor_cost_probe import select_objective_candidate


def test_selection_ignores_validation_and_respects_dollar_metric():
    targets = np.log1p([10., 1000., 50.])
    a = np.log1p([20., 900., 0.])
    b = np.log1p([10., 800., 50.])
    candidates = [(100, a), (200, b)]
    positions = np.array([0, 1])
    assert select_objective_candidate(candidates, targets, positions, 'dollar_mae')[1] == 100
    assert select_objective_candidate(candidates, targets, positions, 'log_mse')[1] == 200
    changed = targets.copy()
    changed[2] = 10
    assert select_objective_candidate(candidates, changed, positions, 'dollar_mae')[1] == 100
