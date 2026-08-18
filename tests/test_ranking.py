import unittest

import numpy as np
import torch

from jepa_models.ranking import (
    PairwiseCostRankingHead,
    all_pair_indices,
    lambdarank_pair_weights,
    pairwise_logistic_loss,
    pairwise_ranknet_loss,
)
from scripts.evaluate_pairwise_cost_ranking import (
    pairwise_accuracy,
    stratified_calibration_split,
    stratified_epoch_batches,
    stratified_fold_assignments,
    top_decile_recall,
)
from scripts.evaluate_locked_pairwise_cost_ranking import apply_saved_isotonic


class TestPairwiseRanking(unittest.TestCase):
    def test_ranknet_loss_backpropagates_through_head(self):
        head = PairwiseCostRankingHead(input_dim=4, hidden_dim=8, dropout=0.0)
        embeddings = torch.randn(6, 4)
        targets = torch.arange(6, dtype=torch.float32)
        scores = head(embeddings)
        partners = torch.roll(torch.arange(6), shifts=1)
        loss = pairwise_ranknet_loss(scores, targets, partners)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in head.parameters()))

    def test_dense_ranknet_uses_every_unique_pair(self):
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
        targets = torch.arange(4, dtype=torch.float32)
        left, right = all_pair_indices(4)
        self.assertEqual(left.numel(), 6)
        loss = pairwise_logistic_loss(scores, targets, left, right)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(scores.grad).all())

    def test_lambdarank_weights_are_finite_and_emphasize_gain_changes(self):
        scores = torch.tensor([0.4, 0.3, 0.2, 0.1])
        relevance = torch.tensor([1.0, 0.2, 0.1, 0.0])
        left = torch.tensor([0, 2])
        right = torch.tensor([3, 3])
        weights = lambdarank_pair_weights(scores, relevance, left, right)
        self.assertTrue(torch.isfinite(weights).all())
        self.assertGreater(weights[0].item(), weights[1].item())

    def test_stratified_batches_cover_each_training_item_once(self):
        targets = np.arange(103, dtype=np.float32)
        batches = stratified_epoch_batches(
            targets,
            batch_size=16,
            num_strata=10,
            seed=42,
            epoch=0,
        )
        observed = np.concatenate(batches)
        self.assertEqual(sorted(observed.tolist()), list(range(103)))
        self.assertTrue(all(len(np.unique(targets[batch] // 11)) > 1 for batch in batches))

    def test_validation_calibration_split_is_disjoint_and_complete(self):
        targets = np.arange(100, dtype=np.float32)
        calibration, scoring = stratified_calibration_split(
            targets,
            fraction=0.5,
            num_strata=10,
            seed=42,
        )
        self.assertFalse(set(calibration) & set(scoring))
        self.assertEqual(set(calibration) | set(scoring), set(range(100)))
        self.assertEqual(len(calibration), 50)

    def test_stratified_fold_assignments_cover_every_fold(self):
        targets = np.arange(100, dtype=np.float32)
        assignments = stratified_fold_assignments(
            targets,
            num_folds=5,
            num_strata=10,
            seed=42,
        )
        self.assertEqual(len(assignments), 100)
        self.assertEqual(set(assignments.tolist()), set(range(5)))
        self.assertTrue(all(np.sum(assignments == fold) == 20 for fold in range(5)))

    def test_heldout_ranking_metrics_have_expected_extremes(self):
        targets = np.arange(20, dtype=np.float32)
        accuracy, evaluated, available = pairwise_accuracy(
            targets,
            targets,
            max_pairs=1000,
            seed=42,
        )
        self.assertEqual(accuracy, 1.0)
        self.assertEqual(evaluated, available)
        self.assertEqual(top_decile_recall(targets, targets), 1.0)

    def test_locked_isotonic_application_clips_both_tails(self):
        scores = np.array([-1.0, 0.5, 3.0])
        calibrated = apply_saved_isotonic(
            scores,
            x_thresholds=np.array([0.0, 1.0, 2.0]),
            y_thresholds=np.array([10.0, 20.0, 40.0]),
        )
        np.testing.assert_allclose(calibrated, [10.0, 15.0, 40.0])


if __name__ == "__main__":
    unittest.main()
