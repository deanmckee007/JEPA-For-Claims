import tempfile
import unittest
import copy
from pathlib import Path

import torch

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import (
    load_claims_model_checkpoint,
    validate_checkpoint_compatibility,
)
from jepa_utils.config import Config


def save_checkpoint_config(path, config):
    torch.save({"hyper_parameters": {"config": config}, "state_dict": {}}, path)


def build_small_model_config(**overrides):
    config = Config()
    config.cpt_vocab_size = 16
    config.icd_vocab_size = 16
    config.ttnc_vocab_size = 12
    config.cpt_rarity_scores = None
    config.icd_rarity_scores = None
    config.ttnc_rarity_scores = None
    config.embedding_dim = 4
    config.output_dim = 4
    config.hidden_dim = 16
    config.ff_hidden_dim = 16
    config.num_layers = 1
    config.num_heads = 2
    config.max_cpt_tokens = 3
    config.max_icd_tokens = 3
    config.max_claims_len = 5
    config.use_sparse_autoencoder = False
    config.use_gated_fusion = False
    config.use_predictor_head = False
    config.use_level1 = True
    config.use_composable_level1 = True
    config.clean_ssl_mode = False
    config.use_token_prediction_head = False
    config.use_diffusion = False
    config.data_contract_hash = "contract-a"
    config.vocab_hash = "vocab-a"
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class TestCheckpointing(unittest.TestCase):
    def test_matching_contract_and_architecture_validate(self):
        config = Config()
        config.data_contract_hash = "contract-a"
        config.vocab_hash = "vocab-a"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.ckpt"
            save_checkpoint_config(path, config)
            saved = validate_checkpoint_compatibility(path, config)

        self.assertEqual(saved.data_contract_hash, "contract-a")

    def test_contract_mismatch_fails_closed(self):
        saved_config = Config()
        saved_config.data_contract_hash = "contract-a"
        saved_config.vocab_hash = "vocab-a"
        current_config = Config()
        current_config.data_contract_hash = "contract-b"
        current_config.vocab_hash = "vocab-a"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.ckpt"
            save_checkpoint_config(path, saved_config)
            with self.assertRaisesRegex(ValueError, "data_contract_hash"):
                validate_checkpoint_compatibility(path, current_config)

    def test_architecture_mismatch_fails_closed(self):
        saved_config = Config()
        saved_config.data_contract_hash = "contract-a"
        saved_config.vocab_hash = "vocab-a"
        current_config = Config()
        current_config.data_contract_hash = "contract-a"
        current_config.vocab_hash = "vocab-a"
        current_config.rnn_type = "transformer"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.ckpt"
            save_checkpoint_config(path, saved_config)
            with self.assertRaisesRegex(ValueError, "rnn_type"):
                validate_checkpoint_compatibility(path, current_config)

    def test_sequence_shape_mismatch_fails_closed(self):
        saved_config = Config()
        saved_config.data_contract_hash = "contract-a"
        saved_config.vocab_hash = "vocab-a"
        current_config = copy.deepcopy(saved_config)
        current_config.max_claims_len += 1

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.ckpt"
            save_checkpoint_config(path, saved_config)
            with self.assertRaisesRegex(ValueError, "max_claims_len"):
                validate_checkpoint_compatibility(path, current_config)

    def test_generator_head_mismatch_requires_explicit_stage_transition(self):
        saved_config = build_small_model_config()
        current_config = copy.deepcopy(saved_config)
        current_config.use_diffusion = True

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.ckpt"
            save_checkpoint_config(path, saved_config)
            with self.assertRaisesRegex(ValueError, "use_diffusion"):
                validate_checkpoint_compatibility(path, current_config)

    def test_stage_transition_loads_encoder_then_syncs_active_embeddings(self):
        saved_config = build_small_model_config(current_stage="stage1")
        saved_model = HierarchicalClaimsModel(saved_config)
        with torch.no_grad():
            saved_model.context_encoder_lvl1.cpt_embedding.weight.fill_(1.25)
            saved_model.context_encoder_lvl1.icd_embedding.weight.fill_(2.25)
            saved_model.context_level1_composer.ttnc_embedding.weight.fill_(3.25)

        current_config = copy.deepcopy(saved_config)
        current_config.current_stage = "stage2"
        current_config.use_token_prediction_head = True
        current_config.use_diffusion = True
        current_config.diffusion_type = "discrete"

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "encoder.ckpt"
            torch.save(
                {
                    "hyper_parameters": {"config": saved_config},
                    "state_dict": saved_model.state_dict(),
                },
                path,
            )
            loaded = load_claims_model_checkpoint(
                HierarchicalClaimsModel,
                path,
                config=current_config,
                stage_transition=True,
            )

        active_embeddings = loaded._get_active_code_embeddings()
        for modality in ("cpt", "icd", "ttnc"):
            self.assertTrue(
                torch.allclose(
                    getattr(loaded.diffusion_model, f"{modality}_embedding").weight,
                    active_embeddings[modality].weight,
                )
            )
        self.assertTrue(loaded.use_token_prediction_head)
        self.assertTrue(loaded.use_diffusion)

    def test_legacy_checkpoint_requires_explicit_override(self):
        saved_config = Config()
        current_config = Config()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.ckpt"
            save_checkpoint_config(path, saved_config)
            with self.assertRaisesRegex(ValueError, "predates"):
                validate_checkpoint_compatibility(path, current_config)
            validate_checkpoint_compatibility(path, current_config, allow_legacy=True)


if __name__ == "__main__":
    unittest.main()
