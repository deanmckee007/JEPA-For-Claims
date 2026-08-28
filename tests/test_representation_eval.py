import unittest

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from jepa_utils.representation_eval import (
    FrozenAttentiveRegressionProbe,
    collect_patient_representations,
    compute_attentive_regression_probe_metrics,
    compute_claim_prototype_metrics,
    compute_claim_prototype_stability,
    compute_representation_geometry_metrics,
    evaluate_representation_quality,
    extract_raw_sequence_lengths,
    extract_last_valid_ttnc,
    get_representation_source_names,
    select_representation_tensor,
)


class TestRepresentationEval(unittest.TestCase):
    def test_collection_exports_frozen_sequences_and_restores_polyak_weights(self):
        class FakeClaimsModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.ones(()))
                self.polyak_active = False

            def activate_eval_polyak_weights(self):
                self.polyak_active = True
                return True

            def restore_online_weights(self):
                self.polyak_active = False
                return True

            def forward(self, cpt_tensor, icd_tensor, ttnc_tensor, target, **kwargs):
                batch_size, sequence_length = ttnc_tensor.shape
                states = torch.ones(batch_size, sequence_length, 4, device=ttnc_tensor.device)
                return {
                    "patient_representation": states.mean(dim=1),
                    "patient_representation_pre_sae": states.mean(dim=1),
                    "sequence_aux": {
                        "sequence_output": states,
                        "valid_token_mask": ttnc_tensor != 0,
                    },
                }

        cpt = torch.ones(3, 2, 1, dtype=torch.long)
        icd = torch.ones(3, 2, 1, dtype=torch.long)
        ttnc = torch.tensor([[0, 1], [2, 3], [0, 4]], dtype=torch.long)
        target = torch.arange(3, dtype=torch.float32)
        dataloader = DataLoader(TensorDataset(cpt, icd, ttnc, target), batch_size=2)
        model = FakeClaimsModel()

        _, _, _, metadata = collect_patient_representations(
            model,
            dataloader,
            device="cpu",
            max_samples=1,
            include_sequence_states=True,
            sequence_state_max_samples=2,
        )

        self.assertEqual(metadata["sequence_states"].shape, (1, 2, 4))
        self.assertEqual(metadata["sequence_state_masks"].shape, (1, 2))
        self.assertEqual(metadata["evaluation_weights"], "polyak")
        self.assertFalse(model.polyak_active)

    def test_frozen_attentive_probe_fits_sequence_signal(self):
        rng = np.random.default_rng(7)
        train_states = rng.normal(size=(48, 4, 8)).astype(np.float32)
        eval_states = rng.normal(size=(16, 4, 8)).astype(np.float32)
        train_masks = np.ones((48, 4), dtype=bool)
        eval_masks = np.ones((16, 4), dtype=bool)
        train_targets = train_states[:, -1, 0] + 0.5 * train_states[:, -1, 1]
        eval_targets = eval_states[:, -1, 0] + 0.5 * eval_states[:, -1, 1]

        metrics = compute_attentive_regression_probe_metrics(
            train_states,
            train_masks,
            train_targets,
            eval_states,
            eval_masks,
            eval_targets,
            device="cpu",
            epochs=3,
            batch_size=16,
            num_heads=4,
            random_state=7,
        )

        self.assertTrue(np.isfinite(metrics["attentive_target_probe_rmse_log1p"]))
        self.assertEqual(metrics["attentive_probe_train_samples"], 48)
        self.assertEqual(metrics["attentive_probe_num_heads"], 4)

    def test_attentive_probe_handles_empty_sequence_mask_without_nan(self):
        probe = FrozenAttentiveRegressionProbe(embedding_dim=4, num_heads=2)
        output = probe(torch.randn(2, 3, 4), torch.zeros(2, 3, dtype=torch.bool))

        self.assertTrue(torch.isfinite(output).all())

    def test_claim_prototype_metrics_and_missing_modality_stability(self):
        assignments = np.eye(4, dtype=np.float32)
        probabilities = assignments * 0.9 + 0.1 / 4

        metrics = compute_claim_prototype_metrics(assignments, probabilities)
        stability = compute_claim_prototype_stability(
            probabilities,
            probabilities.copy(),
        )

        self.assertEqual(metrics["prototype_count"], 4)
        self.assertEqual(metrics["prediction_top1_accuracy"], 1.0)
        self.assertEqual(metrics["prediction_top5_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["effective_prototype_fraction"], 1.0)
        self.assertEqual(stability["top1_agreement"], 1.0)
        self.assertAlmostEqual(stability["jensen_shannon_divergence"], 0.0)

    def test_compute_representation_geometry_metrics_reports_effective_rank(self):
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=np.float32,
        )

        metrics = compute_representation_geometry_metrics(embeddings, cosine_sample_size=4)

        self.assertEqual(metrics["num_samples"], 4)
        self.assertEqual(metrics["embedding_dim"], 3)
        self.assertGreater(metrics["participation_ratio"], 1.0)
        self.assertLess(metrics["participation_ratio"], 3.1)
        self.assertGreaterEqual(metrics["explained_variance_top1_share"], 0.0)
        self.assertLessEqual(metrics["explained_variance_top1_share"], 1.0)
        self.assertEqual(metrics["low_var_frac_lt_1e-3"], 1.0 / 3.0)

    def test_extract_last_valid_ttnc_with_left_padding(self):
        ttnc_tensor = torch.tensor(
            [
                [0, 0, 5, 6],
                [0, 0, 3, 4],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        )

        labels = extract_last_valid_ttnc(ttnc_tensor)

        self.assertEqual(labels.tolist(), [6, 4, 0])

    def test_evaluate_representation_quality_on_separable_embeddings(self):
        embeddings = np.array(
            [
                [1.0, 1.0],
                [1.1, 0.9],
                [0.9, 1.2],
                [1.2, 1.1],
                [-1.0, -1.0],
                [-1.1, -0.9],
                [-0.9, -1.2],
                [-1.2, -1.1],
            ],
            dtype=np.float32,
        )
        specialty_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        targets = np.array([1.0, 1.1, 0.9, 1.2, -1.0, -1.1, -0.9, -1.2], dtype=np.float32)
        raw_sequence_lengths = np.array([2, 2, 3, 3, 8, 9, 10, 12], dtype=np.int64)
        effective_sequence_lengths = np.array([2, 2, 3, 3, 4, 4, 4, 4], dtype=np.int64)

        results = evaluate_representation_quality(
            embeddings,
            specialty_labels,
            targets,
            sequence_lengths=raw_sequence_lengths,
            effective_sequence_lengths=effective_sequence_lengths,
            retrieval_k=3,
        )

        self.assertEqual(results["num_samples"], 8)
        self.assertEqual(results["embedding_dim"], 2)
        self.assertEqual(results["ttnc_proxy_label_source"], "last_valid_ttnc")
        self.assertGreater(results["ttnc_proxy_retrieval_hit_rate_at_3"], 0.99)
        self.assertGreater(results["ttnc_proxy_label_cluster_ari"], 0.99)
        self.assertGreater(results["ttnc_proxy_probe_accuracy"], 0.99)
        self.assertLess(results["target_probe_rmse_log1p"], 0.5)
        self.assertGreater(results["target_probe_rmse_dollars"], 0.0)
        self.assertGreater(results["target_probe_mae_dollars"], 0.0)
        self.assertGreater(results["target_probe_wape_percent"], 0.0)
        self.assertIn("slices", results)
        self.assertIn("target_cost_bucket", results["slices"])
        self.assertIn("sequence_length_bucket", results["slices"])
        self.assertIn("effective_sequence_length_bucket", results["slices"])
        self.assertIn("ttnc_proxy_frequency_bucket", results["slices"])
        self.assertEqual(results["slices"]["sequence_length_bucket_basis"], "raw_claims")

        cost_buckets = results["slices"]["target_cost_bucket"]
        length_buckets = results["slices"]["sequence_length_bucket"]
        effective_length_buckets = results["slices"]["effective_sequence_length_bucket"]
        frequency_buckets = results["slices"]["ttnc_proxy_frequency_bucket"]

        self.assertEqual(sum(bucket["num_samples"] for bucket in cost_buckets.values()), 8)
        self.assertEqual(sum(bucket["num_samples"] for bucket in length_buckets.values()), 8)
        self.assertEqual(sum(bucket["num_samples"] for bucket in effective_length_buckets.values()), 8)
        self.assertEqual(sum(bucket["num_samples"] for bucket in frequency_buckets.values()), 8)
        self.assertLess(
            cost_buckets["q1_low_cost"]["target_dollars_max"],
            cost_buckets["q4_high_cost"]["target_dollars_min"],
        )
        self.assertLessEqual(
            length_buckets["q1_shortest_sequences"]["sequence_length_claims_max"],
            length_buckets["q4_longest_sequences"]["sequence_length_claims_min"],
        )
        self.assertLessEqual(
            effective_length_buckets["q1_shortest_effective_sequences"]["effective_sequence_length_claims_max"],
            effective_length_buckets["q4_longest_effective_sequences"]["effective_sequence_length_claims_min"],
        )
        self.assertLess(
            effective_length_buckets["q4_longest_effective_sequences"]["effective_sequence_length_claims_max"],
            length_buckets["q4_longest_sequences"]["sequence_length_claims_max"],
        )

    def test_extract_raw_sequence_lengths_from_subset(self):
        class TinyDataset:
            def __init__(self):
                self.processed_data = [["a", "b"], ["a", "b", "c", "d"], ["a", "b", "c"]]

            def __len__(self):
                return len(self.processed_data)

            def __getitem__(self, idx):
                return self.processed_data[idx], float(idx)

        subset = Subset(TinyDataset(), [2, 0])
        lengths = extract_raw_sequence_lengths(subset)

        self.assertEqual(lengths.tolist(), [3, 2])

    def test_select_representation_tensor_supports_probe_sources(self):
        outputs = {
            "patient_representation": torch.tensor([[1.0, 2.0]]),
            "patient_representation_pre_sae": torch.tensor([[3.0, 4.0]]),
            "prediction_lvl2": torch.tensor([[[5.0, 6.0], [7.0, 8.0]]]),
            "sequence_aux": {
                "context_mean_pool": torch.tensor([[9.0, 10.0]]),
                "context_max_pool": torch.tensor([[11.0, 12.0]]),
                "context_pooled": torch.tensor([[13.0, 14.0, 15.0, 16.0]]),
                "dense_decoder_latent": torch.tensor([[17.0, 18.0]]),
            },
        }

        self.assertEqual(
            select_representation_tensor(outputs, "patient_representation").tolist(),
            [[1.0, 2.0]],
        )
        self.assertEqual(
            select_representation_tensor(outputs, "patient_representation_pre_sae").tolist(),
            [[3.0, 4.0]],
        )
        self.assertEqual(
            select_representation_tensor(outputs, "next_claim_prediction").tolist(),
            [[7.0, 8.0]],
        )
        self.assertEqual(
            select_representation_tensor(outputs, "context_mean_pool").tolist(),
            [[9.0, 10.0]],
        )
        self.assertIn("dense_decoder_latent", get_representation_source_names())


if __name__ == "__main__":
    unittest.main()
