import unittest
from pathlib import Path

import scripts.evaluate_cost_probe as evaluate_cost_probe
import scripts.evaluate_representations as evaluate_representations
import scripts.run_experiment as run_experiment


class TestExperimentRunner(unittest.TestCase):
    def test_run_experiment_defaults_to_pre_sae_probe_source(self):
        args = run_experiment.parse_args(
            [
                "--name",
                "exp",
                "--recipe",
                "sigreg_dense",
                "--data-path",
                "C:/tmp/fake.parquet",
            ]
        )

        self.assertEqual(args.representation_source, "patient_representation_pre_sae")

    def test_build_train_command_passes_config_overrides(self):
        command = run_experiment.build_train_command(
            name="H1_dense_weight_0p10",
            recipe="sigreg_dense",
            data_path="C:/tmp/fake.parquet",
            checkpoint_path=Path("C:/tmp/encoder.ckpt"),
            accelerator="cpu",
            devices=1,
            seed=123,
            representation_pretrain_epochs=5,
            config_overrides=["observed_claim_loss_weight=0.1", "observed_claim_k=3"],
        )

        self.assertIn("--seed", command)
        self.assertIn("123", command)
        self.assertIn("--set", command)
        self.assertIn("observed_claim_loss_weight=0.1", command)
        self.assertIn("observed_claim_k=3", command)

    def test_build_eval_command_passes_config_overrides(self):
        command = run_experiment.build_eval_command(
            recipe="sigreg_dense",
            data_path="C:/tmp/fake.parquet",
            checkpoint_path=Path("C:/tmp/encoder.ckpt"),
            output_json=Path("C:/tmp/eval.json"),
            accelerator="cpu",
            seed=123,
            max_samples=256,
            retrieval_k=5,
            representation_source="context_mean_pool",
            config_overrides=["observed_claim_loss_weight=0.1"],
        )

        self.assertIn("--seed", command)
        self.assertIn("123", command)
        self.assertIn("--representation-source", command)
        self.assertIn("context_mean_pool", command)
        self.assertIn("--set", command)
        self.assertIn("observed_claim_loss_weight=0.1", command)

    def test_evaluate_script_defaults_to_pre_sae_probe_source(self):
        args = evaluate_representations.parse_args(
            [
                "--checkpoint",
                "C:/tmp/encoder.ckpt",
                "--recipe",
                "sigreg_dense",
            ]
        )

        self.assertEqual(args.representation_source, "patient_representation_pre_sae")

    def test_cost_probe_script_defaults_to_pre_sae_probe_source(self):
        args = evaluate_cost_probe.parse_args(
            [
                "--checkpoint",
                "C:/tmp/encoder.ckpt",
                "--recipe",
                "sigreg_dense",
            ]
        )

        self.assertEqual(args.representation_source, "patient_representation_pre_sae")


if __name__ == "__main__":
    unittest.main()
