import tempfile
import unittest
from pathlib import Path

from scripts.run_recipe_sweep import build_eval_command, build_train_command, write_summary_files


class TestRecipeSweep(unittest.TestCase):
    def test_build_train_command_includes_recipe_and_checkpoint(self):
        checkpoint_path = Path("C:/tmp/vicreg/encoder.ckpt")

        command = build_train_command(
            recipe="vicreg_baseline",
            data_path="C:/tmp/data.parquet",
            checkpoint_path=checkpoint_path,
            accelerator="cpu",
            devices=1,
            representation_pretrain_epochs=3,
        )

        self.assertIn("--recipe", command)
        self.assertIn("vicreg_baseline", command)
        self.assertIn("--out-encoder-ckpt", command)
        self.assertIn(str(checkpoint_path), command)
        self.assertIn("--representation-pretrain-epochs", command)
        self.assertIn("3", command)

    def test_build_eval_command_includes_output_json(self):
        checkpoint_path = Path("C:/tmp/sigreg/encoder.ckpt")
        output_json = Path("C:/tmp/sigreg/eval.json")

        command = build_eval_command(
            recipe="sigreg_dense",
            data_path="C:/tmp/data.parquet",
            checkpoint_path=checkpoint_path,
            output_json=output_json,
            accelerator="auto",
            max_samples=100,
            retrieval_k=7,
            representation_source="next_claim_prediction",
        )

        self.assertIn("--checkpoint", command)
        self.assertIn(str(checkpoint_path), command)
        self.assertIn("--output-json", command)
        self.assertIn(str(output_json), command)
        self.assertIn("--retrieval-k", command)
        self.assertIn("7", command)
        self.assertIn("--representation-source", command)
        self.assertIn("next_claim_prediction", command)

    def test_write_summary_files_emits_json_csv_and_markdown(self):
        results = [
            {
                "recipe": "sigreg_core",
                "checkpoint": "C:/tmp/sigreg_core.ckpt",
                "ttnc_proxy_retrieval_hit_rate_at_5": 0.8,
                "ttnc_proxy_label_cluster_ari": 0.1,
                "ttnc_proxy_probe_accuracy": 0.5,
                "ttnc_proxy_probe_macro_f1": 0.4,
                "target_probe_rmse_log1p": 0.2,
                "target_probe_mae_dollars": 10.0,
                "target_probe_wape_percent": 5.0,
                "target_probe_rmse_dollars": 20.0,
                "cluster_silhouette": 0.3,
                "num_samples": 24,
                "embedding_dim": 256,
                "device": "cpu",
                "ttnc_proxy_label_source": "last_valid_ttnc",
            }
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            summary_json, summary_csv, summary_md = write_summary_files(results, Path(tmp_dir))

            self.assertTrue(summary_json.exists())
            self.assertTrue(summary_csv.exists())
            self.assertTrue(summary_md.exists())
            self.assertIn("sigreg_core", summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
