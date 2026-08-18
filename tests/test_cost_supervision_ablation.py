import unittest

import numpy as np
import torch

from scripts.run_cost_supervision_ablation import (
    DirectCostHead,
    combine_objectives,
    denormalize_cost_predictions,
    stratified_fraction_positions,
    target_normalization,
)
from scripts.evaluate_cost_supervision_artifact import cosine_stability


class TestCostSupervisionAblation(unittest.TestCase):
    def test_stratified_fraction_is_deterministic_and_covers_cost_range(self):
        targets = np.arange(1000, dtype=np.float32)
        first = stratified_fraction_positions(targets, 0.1, seed=42)
        second = stratified_fraction_positions(targets, 0.1, seed=42)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 100)
        selected_deciles = np.unique(targets[first].astype(int) // 100)
        np.testing.assert_array_equal(selected_deciles, np.arange(10))

    def test_full_fraction_selects_every_position(self):
        positions = stratified_fraction_positions(np.arange(17), 1.0, seed=42)
        np.testing.assert_array_equal(positions, np.arange(17))

    def test_target_normalization_round_trips(self):
        targets = np.array([1.0, 2.0, 4.0], dtype=np.float32)
        mean, std = target_normalization(targets)
        normalized = (targets - mean) / std
        reconstructed = denormalize_cost_predictions(normalized, mean, std)
        np.testing.assert_allclose(reconstructed, targets, rtol=1e-6)

    def test_task_only_objective_excludes_ssl(self):
        ssl_loss = torch.tensor(10.0)
        task_loss = torch.tensor(2.0)
        self.assertEqual(
            combine_objectives(ssl_loss, task_loss, "supervised_scratch", 0.25),
            task_loss,
        )
        self.assertEqual(
            combine_objectives(ssl_loss, task_loss, "ssl_finetune", 0.25),
            task_loss,
        )
        self.assertEqual(
            combine_objectives(ssl_loss, task_loss, "joint_ssl_cost", 0.25),
            torch.tensor(10.5),
        )

    def test_direct_cost_head_shape(self):
        head = DirectCostHead(input_dim=8, hidden_dim=16)
        self.assertEqual(head(torch.randn(5, 8)).shape, (5,))

    def test_cosine_stability_is_one_for_identical_embeddings(self):
        embeddings = np.eye(3, dtype=np.float32)
        metrics = cosine_stability(embeddings, embeddings)
        self.assertAlmostEqual(metrics["full_to_missing_cosine_mean"], 1.0)
        self.assertAlmostEqual(metrics["full_to_missing_cosine_median"], 1.0)


if __name__ == "__main__":
    unittest.main()
