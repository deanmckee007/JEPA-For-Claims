import unittest
from pathlib import Path
from types import SimpleNamespace

import scripts.train as train
from jepa_utils.config import (
    Config,
    apply_config_overrides,
    apply_runtime_config_overrides,
    apply_training_recipe,
    get_training_recipe_names,
    parse_config_overrides,
)


class TestTrainingRecipes(unittest.TestCase):
    def test_lejepa_sigreg_requires_a_convex_lambda(self):
        config = Config()
        config.sigreg_formulation = "lejepa_convex"
        config.sigreg_weight_lvl2 = 1.01

        with self.assertRaisesRegex(ValueError, "sigreg_weight_lvl2"):
            apply_runtime_config_overrides(config)

    def test_recipe_catalog_contains_expected_names(self):
        names = get_training_recipe_names()

        self.assertIn("custom", names)
        self.assertIn("vicreg_baseline", names)
        self.assertIn("sigreg_core", names)
        self.assertIn("sigreg_dense", names)
        self.assertIn("sigreg_dense_hybrid", names)
        self.assertIn("sigreg_dense_hybrid_repr", names)
        self.assertIn("sigreg_dense_hybrid_dollar", names)
        self.assertIn("sigreg_dense_hybrid_dollar_bifurcated_state", names)
        self.assertIn("sigreg_dense_hybrid_dollar_bifurcated_state_bottleneck128", names)
        self.assertIn("sigreg_dense_hybrid_dollar_bifurcated_state_residual", names)
        self.assertIn("sigreg_dense_hybrid_dollar_masked_grounding", names)
        self.assertIn("sigreg_dense_hybrid_dollar_masked_grounding_sigreg0", names)
        self.assertIn("sigreg_dense_hybrid_dollar_masked_grounding_multihorizon", names)
        self.assertIn("a9_world_model_claims", names)
        self.assertIn("a10_temporal_contrastive", names)
        self.assertIn("a11_graph_ssl_transfer", names)
        self.assertIn("sigreg_dense_hybrid_dollar_cost_head", names)
        self.assertIn("composable_level1_sigreg_any_code", names)
        self.assertIn("composable_level1_lejepa", names)
        self.assertIn("composable_level1_lejepa_any_code", names)
        self.assertIn("composable_level1_capi", names)
        self.assertIn("levjepa_patient_views", names)

    def test_composable_lejepa_recipe_removes_ema_and_enables_level2_sigreg(self):
        config = apply_training_recipe(Config(), "composable_level1_lejepa")
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.sigreg_formulation, "lejepa_convex")
        self.assertEqual(config.sigreg_weight_lvl2, 0.05)
        self.assertEqual(config.target_encoder_mode_lvl1, "shared")
        self.assertEqual(config.target_encoder_mode_lvl2, "shared")
        self.assertTrue(config.use_composable_level1)

    def test_composable_capi_recipe_keeps_no_ema_continuous_anchor(self):
        config = apply_training_recipe(Config(), "composable_level1_capi")
        config = apply_runtime_config_overrides(config)

        self.assertTrue(config.use_claim_prototypes)
        self.assertEqual(config.claim_prototype_count, 64)
        self.assertEqual(config.claim_prototype_weight, 0.05)
        self.assertEqual(config.target_encoder_mode_lvl1, "shared")
        self.assertEqual(config.target_encoder_mode_lvl2, "shared")
        self.assertEqual(config.sigreg_formulation, "lejepa_convex")

    def test_composable_any_code_recipe_retains_partial_claims(self):
        config = apply_training_recipe(
            Config(),
            "composable_level1_sigreg_any_code",
        )
        self.assertTrue(config.use_composable_level1)
        self.assertEqual(config.level1_marginal_regularizer, "sigreg")
        self.assertEqual(config.claim_inclusion_policy, "any_code")

    def test_selected_composable_lejepa_recipe_uses_light_level1_objective(self):
        config = apply_training_recipe(
            Config(),
            "composable_level1_lejepa_any_code",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.claim_inclusion_policy, "any_code")
        self.assertEqual(config.level1_predictive_weight, 0.1)
        self.assertEqual(config.target_encoder_mode_lvl1, "shared")
        self.assertEqual(config.target_encoder_mode_lvl2, "shared")
        self.assertTrue(config.use_composable_level1)

    def test_levjepa_patient_view_recipe_is_additive_shared_and_projected(self):
        config = apply_training_recipe(Config(), "levjepa_patient_views")
        config = apply_runtime_config_overrides(config)

        self.assertTrue(config.use_levjepa_patient_views)
        self.assertEqual(config.sigreg_formulation, "levjepa_additive")
        self.assertEqual(config.sigreg_weight_lvl2, 0.02)
        self.assertEqual(config.sigreg_num_slices, 1024)
        self.assertEqual(config.target_encoder_mode_lvl1, "shared")
        self.assertEqual(config.target_encoder_mode_lvl2, "shared")
        self.assertEqual(config.levjepa_num_local_views, 2)
        self.assertEqual(config.levjepa_claim_drop_ratio, 0.3)
        self.assertFalse(config.use_level2_dense_prediction)
        self.assertFalse(config.use_masked_next_claim_token_grounding)
        self.assertFalse(config.use_sparse_autoencoder)
        self.assertTrue(config.use_eval_polyak_average)
        self.assertEqual(config.eval_polyak_decay, 0.9)
        self.assertEqual(config.eval_polyak_update_interval, 1)

    def test_levjepa_patient_views_require_paper_faithful_formulation(self):
        config = Config()
        config.use_levjepa_patient_views = True
        config.ssl_objective_type = "sigreg"
        config.target_encoder_mode = "shared"
        config.sigreg_formulation = "lejepa_convex"

        with self.assertRaisesRegex(ValueError, "levjepa_additive"):
            apply_runtime_config_overrides(config)

    def test_levjepa_patient_views_reprelu_changes_only_canonical_link(self):
        dense = apply_runtime_config_overrides(
            apply_training_recipe(Config(), "levjepa_patient_views")
        )
        sparse = apply_runtime_config_overrides(
            apply_training_recipe(Config(), "levjepa_patient_views_reprelu")
        )

        self.assertEqual(dense.representation_link_lvl2, "identity")
        self.assertEqual(sparse.representation_link_lvl2, "reprelu")
        self.assertTrue(sparse.use_levjepa_patient_views)
        self.assertEqual(sparse.sigreg_formulation, dense.sigreg_formulation)
        self.assertEqual(sparse.sigreg_weight_lvl2, dense.sigreg_weight_lvl2)
        self.assertEqual(sparse.levjepa_claim_drop_ratio, dense.levjepa_claim_drop_ratio)

    def test_sigreg_dense_recipe_applies_expected_ssl_flags(self):
        config = apply_training_recipe(Config(), "sigreg_dense")
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.train_recipe, "sigreg_dense")
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "shared")
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertEqual(config.observed_claim_k, 2)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_sigreg_dense_hybrid_recipe_applies_expected_ssl_flags(self):
        config = apply_training_recipe(Config(), "sigreg_dense_hybrid")
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.train_recipe, "sigreg_dense_hybrid")
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.target_encoder_mode_lvl1, "shared")
        self.assertEqual(config.target_encoder_mode_lvl2, "ema")
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertEqual(config.observed_claim_k, 2)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_sigreg_dense_hybrid_repr_recipe_applies_expected_schedule(self):
        config = apply_training_recipe(Config(), "sigreg_dense_hybrid_repr")
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.train_recipe, "sigreg_dense_hybrid_repr")
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.representation_pretrain_epochs, 20)
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_sigreg_dense_hybrid_dollar_recipe_applies_expected_schedule(self):
        config = apply_training_recipe(Config(), "sigreg_dense_hybrid_dollar")
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.train_recipe, "sigreg_dense_hybrid_dollar")
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.representation_pretrain_epochs, 10)
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_bifurcated_state_recipe_enables_separate_predictive_path(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_bifurcated_state",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(
            config.train_recipe,
            "sigreg_dense_hybrid_dollar_bifurcated_state",
        )
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.representation_pretrain_epochs, 10)
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertTrue(config.use_bifurcated_patient_state)
        self.assertFalse(config.use_predictor_head)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_bifurcated_state_residual_recipe_enables_mean_residual_path(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_bifurcated_state_residual",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(
            config.train_recipe,
            "sigreg_dense_hybrid_dollar_bifurcated_state_residual",
        )
        self.assertTrue(config.use_bifurcated_patient_state)
        self.assertTrue(config.use_patient_state_mean_residual)
        self.assertEqual(config.patient_state_mean_residual_weight, 0.5)
        self.assertFalse(config.use_predictor_head)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_bifurcated_state_bottleneck_recipe_narrows_predictive_branch(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_bifurcated_state_bottleneck128",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(
            config.train_recipe,
            "sigreg_dense_hybrid_dollar_bifurcated_state_bottleneck128",
        )
        self.assertTrue(config.use_bifurcated_patient_state)
        self.assertTrue(config.use_predictive_state_bottleneck)
        self.assertEqual(config.predictive_state_bottleneck_dim, 128)
        self.assertFalse(config.use_patient_state_mean_residual)
        self.assertFalse(config.use_predictor_head)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_masked_grounding_recipe_enables_ordered_next_claim_aux_loss(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_masked_grounding",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(
            config.train_recipe,
            "sigreg_dense_hybrid_dollar_masked_grounding",
        )
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.representation_pretrain_epochs, 10)
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertTrue(config.use_masked_next_claim_token_grounding)
        self.assertEqual(config.masked_next_claim_mask_ratio, 0.5)
        self.assertEqual(config.masked_next_claim_token_weight, 0.2)
        self.assertTrue(config.masked_next_claim_use_cpt)
        self.assertTrue(config.masked_next_claim_use_icd)
        self.assertTrue(config.masked_next_claim_sort_target_tokens)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_masked_grounding_sigreg0_recipe_matches_current_best_overrides(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_masked_grounding_sigreg0",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(
            config.train_recipe,
            "sigreg_dense_hybrid_dollar_masked_grounding_sigreg0",
        )
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.representation_pretrain_epochs, 10)
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertEqual(config.observed_claim_k, 2)
        self.assertEqual(config.observed_claim_loss_weight, 0.0)
        self.assertEqual(config.masked_next_claim_mask_ratio, 0.7)
        self.assertEqual(config.masked_next_claim_token_weight, 0.2)
        self.assertEqual(config.sigreg_weight_lvl2, 0.0)
        self.assertTrue(config.use_masked_next_claim_token_grounding)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_a9_world_model_recipe_enables_latent_dynamics_branch(self):
        config = apply_training_recipe(
            Config(),
            "a9_world_model_claims",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.train_recipe, "a9_world_model_claims")
        self.assertTrue(config.use_world_model_dynamics)
        self.assertEqual(config.future_claim_k, 2)
        self.assertEqual(config.world_model_future_steps, 3)
        self.assertEqual(config.world_model_summary_weight, 0.5)
        self.assertEqual(config.sigreg_weight_lvl2, 0.0)
        self.assertTrue(config.use_masked_next_claim_token_grounding)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_a10_temporal_contrastive_recipe_enables_future_contrastive_branch(self):
        config = apply_training_recipe(
            Config(),
            "a10_temporal_contrastive",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.train_recipe, "a10_temporal_contrastive")
        self.assertTrue(config.use_temporal_contrastive)
        self.assertEqual(config.temporal_ssl_mode, "cpc")
        self.assertEqual(config.temporal_loss_weight, 0.1)
        self.assertEqual(config.temporal_context_k, 4)
        self.assertEqual(config.temporal_future_steps, 3)
        self.assertEqual(config.future_claim_k, 2)
        self.assertEqual(config.observed_claim_loss_weight, 0.0)
        self.assertEqual(config.sigreg_weight_lvl2, 0.0)
        self.assertTrue(config.use_masked_next_claim_token_grounding)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_a11_graph_ssl_transfer_recipe_enables_graph_embedding_transfer(self):
        config = apply_training_recipe(
            Config(),
            "a11_graph_ssl_transfer",
        )
        config = apply_config_overrides(
            config,
            overrides={"graph_embedding_path": "tmp/fake_graph_embeddings.pt"},
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.train_recipe, "a11_graph_ssl_transfer")
        self.assertTrue(config.use_graph_pretrained_code_embeddings)
        self.assertEqual(config.graph_embedding_path, "tmp/fake_graph_embeddings.pt")
        self.assertEqual(config.graph_embedding_mix, 1.0)
        self.assertTrue(config.graph_transfer_cpt)
        self.assertTrue(config.graph_transfer_icd)
        self.assertTrue(config.graph_transfer_ttnc)
        self.assertEqual(config.representation_pretrain_epochs, 24)
        self.assertTrue(config.use_masked_next_claim_token_grounding)
        self.assertEqual(config.sigreg_weight_lvl2, 0.0)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_masked_grounding_multihorizon_recipe_enables_future_claim_targets(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_masked_grounding_multihorizon",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(
            config.train_recipe,
            "sigreg_dense_hybrid_dollar_masked_grounding_multihorizon",
        )
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.representation_pretrain_epochs, 10)
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertEqual(config.observed_claim_k, 2)
        self.assertEqual(config.observed_claim_loss_weight, 0.0)
        self.assertEqual(config.future_claim_k, 2)
        self.assertEqual(config.future_claim_loss_weight, 0.5)
        self.assertEqual(config.future_claim_loss_decay, 0.5)
        self.assertEqual(config.next_claim_loss_weight, 1.0)
        self.assertTrue(config.use_masked_next_claim_token_grounding)
        self.assertEqual(config.masked_next_claim_mask_ratio, 0.7)
        self.assertEqual(config.masked_next_claim_token_weight, 0.2)
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_masked_grounding_requires_at_least_one_supervised_stream(self):
        config = Config()
        config.use_masked_next_claim_token_grounding = True
        config.masked_next_claim_use_cpt = False
        config.masked_next_claim_use_icd = False
        config.masked_next_claim_include_ttnc = False

        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(config)

    def test_future_claim_targets_require_dense_prediction(self):
        config = Config()
        config.future_claim_k = 2
        config.use_level2_dense_prediction = False

        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(config)

    def test_stage1_builder_disables_token_head_by_default(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_masked_grounding",
        )
        config = apply_config_overrides(
            config,
            overrides={
                "clean_ssl_mode": False,
                "use_token_prediction_head": True,
                "use_diffusion": False,
            },
        )
        config = apply_runtime_config_overrides(config)

        stage_cfg = train.build_stage1_config(config)

        self.assertFalse(stage_cfg.use_token_prediction_head)
        self.assertFalse(stage_cfg.use_diffusion)

    def test_stage1_builder_can_explicitly_keep_token_head(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_masked_grounding",
        )
        config = apply_config_overrides(
            config,
            overrides={
                "clean_ssl_mode": False,
                "use_token_prediction_head": True,
                "use_diffusion": False,
                "allow_stage1_token_prediction_head": True,
            },
        )
        config = apply_runtime_config_overrides(config)

        stage_cfg = train.build_stage1_config(config)

        self.assertTrue(stage_cfg.use_token_prediction_head)
        self.assertFalse(stage_cfg.use_diffusion)

    def test_cost_head_recipe_enables_predictor_head_with_explicit_weight(self):
        config = apply_training_recipe(
            Config(),
            "sigreg_dense_hybrid_dollar_cost_head",
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(
            config.train_recipe,
            "sigreg_dense_hybrid_dollar_cost_head",
        )
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.representation_pretrain_epochs, 10)
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertTrue(config.use_predictor_head)
        self.assertEqual(config.task_loss_weight, 0.25)
        self.assertEqual(config.predictor_head_source, "context_mean_pool")
        self.assertFalse(config.use_token_prediction_head)
        self.assertFalse(config.use_diffusion)

    def test_predictor_head_source_is_validated(self):
        config = Config()
        config.use_predictor_head = True
        config.predictor_head_source = "not_a_real_source"
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(config)

    def test_optimizer_and_scheduler_overrides_are_applied_and_normalized(self):
        config = apply_config_overrides(
            Config(),
            overrides={
                "optimizer_type": "Adam",
                "weight_decay": 0.0,
                "scheduler_type": "Cosine",
                "scheduler_warmup_epochs": 1,
                "train_batch_size": 256,
                "eval_batch_size": 64,
                "checkpoint_save_top_k": 2,
                "checkpoint_every_n_epochs": 1,
                "checkpoint_mode": "Max",
            },
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.optimizer_type, "adam")
        self.assertEqual(config.weight_decay, 0.0)
        self.assertEqual(config.scheduler_type, "cosine")
        self.assertEqual(config.scheduler_warmup_epochs, 1)
        self.assertEqual(config.train_batch_size, 256)
        self.assertEqual(config.eval_batch_size, 64)
        self.assertEqual(config.checkpoint_save_top_k, 2)
        self.assertEqual(config.checkpoint_every_n_epochs, 1)
        self.assertEqual(config.checkpoint_mode, "max")

    def test_invalid_optimizer_and_batch_size_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(Config(optimizer_type="not_real"))
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(Config(train_batch_size=0))
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(Config(checkpoint_mode="sideways"))
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(Config(freeze_logvars_after_epoch=-1))
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(Config(checkpoint_dirpath=""))
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(Config(temporal_ssl_mode="not_real"))
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(
                Config(use_graph_pretrained_code_embeddings=True)
            )
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(
                Config(
                    use_graph_pretrained_code_embeddings=True,
                    graph_embedding_path="tmp/fake.pt",
                    graph_embedding_mix=1.5,
                )
            )
        with self.assertRaises(ValueError):
            apply_runtime_config_overrides(
                Config(
                    use_graph_pretrained_code_embeddings=True,
                    graph_embedding_path="tmp/fake.pt",
                    graph_transfer_cpt=False,
                    graph_transfer_icd=False,
                    graph_transfer_ttnc=False,
                )
            )

    def test_build_stage_callbacks_support_best_last_and_periodic_checkpoints(self):
        config = Config(
            checkpoint_save_top_k=2,
            checkpoint_save_last=True,
            checkpoint_every_n_epochs=1,
            checkpoint_dirpath="tmp/test_stage_callbacks",
        )
        config.use_predictor_head = True
        config = apply_runtime_config_overrides(config)

        callbacks, monitor, mode = train.build_stage_callbacks(config, "stage1")
        checkpoint_callbacks = [
            callback
            for callback in callbacks
            if callback.__class__.__name__ == "ModelCheckpoint"
        ]

        self.assertEqual(monitor, "loss")
        self.assertEqual(mode, "min")
        self.assertEqual(len(checkpoint_callbacks), 2)
        self.assertEqual(checkpoint_callbacks[0].save_top_k, 2)
        self.assertTrue(checkpoint_callbacks[0].save_last)
        self.assertEqual(Path(checkpoint_callbacks[0].dirpath).name, "test_stage_callbacks")
        self.assertEqual(checkpoint_callbacks[1].every_n_epochs, 1)
        self.assertEqual(Path(checkpoint_callbacks[1].dirpath).name, "test_stage_callbacks")

    def test_build_config_from_args_applies_recipe_and_cli_overrides(self):
        args = SimpleNamespace(
            recipe="sigreg_core",
            data_path="C:/tmp/fake.parquet",
            accelerator="cpu",
            devices=1,
            representation_pretrain_epochs=3,
            generator_train_epochs=0,
            joint_train_epochs=0,
            out_encoder_ckpt="enc.ckpt",
            pretrained_encoder_ckpt="prev.ckpt",
            seed=123,
            clean_ssl_mode=False,
            disable_generative_save=True,
            config_overrides=[
                "observed_claim_k=3",
                "use_level2_dense_prediction=True",
                "train_batch_size=256",
                "optimizer_type=adam",
            ],
        )

        config = train.build_config_from_args(args)

        self.assertEqual(config.train_recipe, "sigreg_core")
        self.assertEqual(config.data_path, "C:/tmp/fake.parquet")
        self.assertEqual(config.trainer_accelerator, "cpu")
        self.assertEqual(config.trainer_devices, 1)
        self.assertEqual(config.representation_pretrain_epochs, 3)
        self.assertEqual(config.out_encoder_ckpt, "enc.ckpt")
        self.assertEqual(config.pretrained_encoder_ckpt, "prev.ckpt")
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.ssl_objective_type, "sigreg")
        self.assertEqual(config.target_encoder_mode, "shared")
        self.assertFalse(config.use_generative_save)
        self.assertEqual(config.observed_claim_k, 3)
        self.assertTrue(config.use_level2_dense_prediction)
        self.assertEqual(config.train_batch_size, 256)
        self.assertEqual(config.optimizer_type, "adam")

    def test_cli_epoch_override_can_replace_recipe_schedule(self):
        args = SimpleNamespace(
            recipe="sigreg_dense_hybrid_dollar",
            data_path=None,
            accelerator=None,
            devices=None,
            representation_pretrain_epochs=12,
            generator_train_epochs=None,
            joint_train_epochs=None,
            out_encoder_ckpt=None,
            pretrained_encoder_ckpt=None,
            seed=None,
            clean_ssl_mode=False,
            disable_generative_save=False,
            config_overrides=None,
        )

        config = train.build_config_from_args(args)

        self.assertEqual(config.train_recipe, "sigreg_dense_hybrid_dollar")
        self.assertEqual(config.representation_pretrain_epochs, 12)

    def test_parse_and_apply_config_overrides(self):
        parsed = parse_config_overrides(
            ["observed_claim_loss_weight=0.1", "use_level2_dense_prediction=True"]
        )
        self.assertEqual(parsed["observed_claim_loss_weight"], 0.1)
        self.assertTrue(parsed["use_level2_dense_prediction"])

        config = apply_config_overrides(Config(), overrides=parsed)
        self.assertEqual(config.observed_claim_loss_weight, 0.1)
        self.assertTrue(config.use_level2_dense_prediction)

    def test_hybrid_target_encoder_resolves_per_level_defaults(self):
        config = apply_runtime_config_overrides(
            Config(target_encoder_mode="hybrid")
        )

        self.assertEqual(config.target_encoder_mode, "hybrid")
        self.assertEqual(config.target_encoder_mode_lvl1, "shared")
        self.assertEqual(config.target_encoder_mode_lvl2, "ema")

    def test_embedding_override_refreshes_derived_dims(self):
        config = apply_config_overrides(Config(), overrides={"embedding_dim": 64})
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.embedding_dim, 64)
        self.assertEqual(config.output_dim, 64)
        self.assertEqual(config.patient_representation_dim, 128)

    def test_explicit_output_dim_override_is_preserved(self):
        config = apply_config_overrides(
            Config(),
            overrides={
                "embedding_dim": 64,
                "output_dim": 96,
                "use_context_pooled_patient_representation": False,
            },
        )
        config = apply_runtime_config_overrides(config)

        self.assertEqual(config.embedding_dim, 64)
        self.assertEqual(config.output_dim, 96)
        self.assertEqual(config.patient_representation_dim, 64)


if __name__ == "__main__":
    unittest.main()
