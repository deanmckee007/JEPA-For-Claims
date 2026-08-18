import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import torch

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_models.encoders import Level1Encoder
from jepa_models.claim_prototypes import ClaimPrototypeObjective, sinkhorn_assignments
from jepa_models.prediction_blocks import Level2PredictionBlock
from jepa_models.ssl_objectives import SIGRegObjective, WristbandGaussianRegularizer
from jepa_utils.config import Config


def build_config(**overrides):
    cfg = Config()
    cfg.cpt_vocab_size = 16
    cfg.icd_vocab_size = 16
    cfg.ttnc_vocab_size = 12
    cfg.cpt_rarity_scores = None
    cfg.icd_rarity_scores = None
    cfg.ttnc_rarity_scores = None
    cfg.embedding_dim = 4
    cfg.output_dim = cfg.embedding_dim
    cfg.hidden_dim = 16
    cfg.ff_hidden_dim = 16
    cfg.num_layers = 1
    cfg.num_heads = 2
    cfg.max_cpt_tokens = 3
    cfg.max_icd_tokens = 3
    cfg.max_claims_len = 5
    cfg.use_sparse_autoencoder = False
    cfg.use_gated_fusion = False
    cfg.use_diffusion = False
    cfg.use_token_prediction_head = False
    cfg.use_predictor_head = False
    cfg.use_level1 = False
    cfg.dropout = 0.0
    cfg.representation_pretrain_epochs = 0
    cfg.generator_train_epochs = 0
    cfg.joint_train_epochs = 0
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def make_batch(cfg, batch_size=2):
    torch.manual_seed(0)
    cpt_tensor = torch.randint(
        1,
        cfg.cpt_vocab_size,
        (batch_size, cfg.max_claims_len, cfg.max_cpt_tokens),
    )
    icd_tensor = torch.randint(
        1,
        cfg.icd_vocab_size,
        (batch_size, cfg.max_claims_len, cfg.max_icd_tokens),
    )
    ttnc_tensor = torch.randint(
        1,
        cfg.ttnc_vocab_size,
        (batch_size, cfg.max_claims_len),
    )
    target = torch.randn(batch_size)
    return cpt_tensor, icd_tensor, ttnc_tensor, target


class TestSSLModernization(unittest.TestCase):
    def test_positionwise_sinkhorn_balances_each_claim_slot(self):
        torch.manual_seed(7)
        assignments = sinkhorn_assignments(torch.randn(32, 8), iterations=5)

        self.assertTrue(
            torch.allclose(
                assignments.sum(dim=1),
                torch.ones(32),
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                assignments.sum(dim=0),
                torch.full((8,), 4.0),
                atol=1e-3,
            )
        )

    def test_claim_prototype_target_representation_is_detached(self):
        objective = ClaimPrototypeObjective(
            embedding_dim=4,
            num_prototypes=4,
            sinkhorn_iterations=4,
        )
        prediction = torch.randn(16, 1, 4, requires_grad=True)
        target = torch.randn(16, 1, 4, requires_grad=True)
        metrics = objective(
            prediction,
            target,
            mask=torch.ones(16, 1, dtype=torch.bool),
        )

        metrics["total"].backward()

        self.assertIsNotNone(prediction.grad)
        self.assertIsNone(target.grad)
        self.assertIsNotNone(objective.student_head.weight.grad)
        self.assertIsNotNone(objective.clustering_head.weight.grad)
        self.assertGreater(metrics["effective_prototypes"].item(), 1.0)

    def test_composed_claim_prototype_auxiliary_uses_only_weighted_slots(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            sigreg_formulation="lejepa_convex",
            sigreg_weight_lvl1=0.0,
            sigreg_weight_lvl2=0.05,
            target_encoder_mode="shared",
            use_level1=True,
            use_composable_level1=True,
            level1_marginal_regularizer="sigreg",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            observed_claim_loss_weight=0.0,
            next_claim_loss_weight=1.0,
            use_claim_prototypes=True,
            claim_prototype_count=4,
            claim_prototype_weight=0.05,
        )
        model = HierarchicalClaimsModel(cfg)
        cpt, icd, ttnc, target = make_batch(cfg, batch_size=8)

        outputs = model(cpt, icd, ttnc, target=target)

        self.assertTrue(torch.isfinite(outputs["loss"]).item())
        self.assertGreater(outputs["claim_prototype_loss"].item(), 0.0)
        self.assertEqual(
            int(outputs["claim_prototype_active_mask"].sum().item()),
            8,
        )
        self.assertEqual(
            outputs["claim_prototype_assignments"].shape,
            (8, 3, 4),
        )
        self.assertEqual(
            model.representation_sharing_contract()["claim_prototype_target"],
            "detached_shared_composed_claim",
        )

    def test_claim_prototype_initialization_does_not_shift_global_rng(self):
        shared = dict(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_level1=True,
            use_composable_level1=True,
        )
        torch.manual_seed(314)
        HierarchicalClaimsModel(build_config(**shared))
        baseline_tail = torch.rand(8)

        torch.manual_seed(314)
        HierarchicalClaimsModel(
            build_config(
                **shared,
                use_claim_prototypes=True,
                claim_prototype_count=16,
            )
        )
        prototype_tail = torch.rand(8)

        self.assertTrue(torch.equal(baseline_tail, prototype_tail))

    def test_context_modality_dropout_never_erases_both_modalities(self):
        cfg = build_config(
            context_cpt_dropout_prob=0.5,
            context_icd_dropout_prob=0.5,
        )
        model = HierarchicalClaimsModel(cfg)
        model.train()
        cpt, icd, ttnc, _ = make_batch(cfg)

        with patch("torch.rand", side_effect=lambda shape, device=None: torch.zeros(shape, device=device)):
            dropped_cpt, dropped_icd = model._apply_context_modality_dropout(
                cpt, icd, ttnc
            )

        self.assertFalse(dropped_cpt.ne(0).any())
        self.assertTrue(dropped_icd.ne(0).all())
        self.assertFalse(
            (~dropped_cpt.ne(0).any(dim=-1) & ~dropped_icd.ne(0).any(dim=-1)).any()
        )

    def test_context_modality_dropout_preserves_natural_single_modality_claim(self):
        cfg = build_config(
            context_cpt_dropout_prob=0.5,
            context_icd_dropout_prob=0.5,
        )
        model = HierarchicalClaimsModel(cfg)
        model.train()
        cpt, icd, ttnc, _ = make_batch(cfg)
        cpt[:, 0] = 0
        original_icd = icd[:, 0].clone()

        with patch("torch.rand", side_effect=lambda shape, device=None: torch.zeros(shape, device=device)):
            dropped_cpt, dropped_icd = model._apply_context_modality_dropout(
                cpt, icd, ttnc
            )

        self.assertFalse(dropped_cpt[:, 0].ne(0).any())
        self.assertTrue(torch.equal(dropped_icd[:, 0], original_icd))

    def test_composable_level1_is_the_level2_input_and_supports_missing_modalities(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_level1=True,
            use_composable_level1=True,
            level1_marginal_regularizer="none",
            sigreg_weight_lvl1=0.0,
            sigreg_weight_lvl2=0.0,
        )
        model = HierarchicalClaimsModel(cfg)
        cpt, icd, ttnc, target = make_batch(cfg)

        full, components = model.encode_claims(
            cpt, icd, ttnc, return_components=True
        )
        missing_cpt = model.encode_claims(torch.zeros_like(cpt), icd, ttnc)
        missing_icd = model.encode_claims(cpt, torch.zeros_like(icd), ttnc)

        self.assertEqual(full.shape, (2, cfg.max_claims_len, cfg.embedding_dim))
        self.assertIn("shared_interaction", components)
        self.assertTrue(torch.isfinite(missing_cpt).all())
        self.assertTrue(torch.isfinite(missing_icd).all())
        self.assertFalse(torch.allclose(full, missing_cpt))
        self.assertFalse(torch.allclose(full, missing_icd))

        outputs = model.training_forward(cpt, icd, ttnc, target)
        outputs["loss"].backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.context_encoder_lvl1.parameters()
        ))
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.context_level1_composer.parameters()
        ))

    def test_composable_representation_sharing_contract_reports_effective_modules(self):
        cfg = build_config(
            use_level1=True,
            use_composable_level1=True,
            target_encoder_mode="hybrid",
            target_encoder_mode_lvl1="shared",
            target_encoder_mode_lvl2="ema",
        )
        model = HierarchicalClaimsModel(cfg)

        contract = model.representation_sharing_contract()

        self.assertEqual(
            contract["level2_claim_input"],
            "composed_level1_claim_state",
        )
        self.assertTrue(contract["level1_task_and_level2_input_encoder_shared"])
        self.assertEqual(contract["context_target_level1_encoder_mode"], "shared")
        self.assertEqual(contract["context_target_claim_composer_mode"], "ema")
        self.assertEqual(contract["legacy_level2_token_encoder_mode"], "inactive")
        self.assertIs(model._get_ssl_target_encoder_lvl1(), model.context_encoder_lvl1)
        self.assertIs(model._get_ssl_target_composer(), model.target_level1_composer)

    def test_fully_shared_composable_path_reuses_context_encoder_and_composer(self):
        cfg = build_config(
            use_level1=True,
            use_composable_level1=True,
            target_encoder_mode="shared",
            target_encoder_mode_lvl1="shared",
            target_encoder_mode_lvl2="shared",
        )
        model = HierarchicalClaimsModel(cfg)

        contract = model.representation_sharing_contract()

        self.assertEqual(contract["context_target_level1_encoder_mode"], "shared")
        self.assertEqual(contract["context_target_claim_composer_mode"], "shared")
        self.assertIs(model._get_ssl_target_encoder_lvl1(), model.context_encoder_lvl1)
        self.assertIs(model._get_ssl_target_composer(), model.context_level1_composer)

    def test_composable_generator_uses_and_resynchronizes_active_embeddings(self):
        cfg = build_config(
            use_level1=True,
            use_composable_level1=True,
            use_diffusion=True,
            diffusion_type="discrete",
            clean_ssl_mode=False,
        )
        model = HierarchicalClaimsModel(cfg)
        active_embeddings = model._get_active_code_embeddings()

        self.assertIs(active_embeddings["cpt"], model.context_encoder_lvl1.cpt_embedding)
        self.assertIs(active_embeddings["icd"], model.context_encoder_lvl1.icd_embedding)
        self.assertIs(active_embeddings["ttnc"], model.context_level1_composer.ttnc_embedding)

        with torch.no_grad():
            active_embeddings["cpt"].weight.fill_(1.5)
            active_embeddings["icd"].weight.fill_(2.5)
            active_embeddings["ttnc"].weight.fill_(3.5)
        model.synchronize_generator_embeddings_from_active_encoder()

        for modality in ("cpt", "icd", "ttnc"):
            self.assertTrue(
                torch.allclose(
                    getattr(model.diffusion_model, f"{modality}_embedding").weight,
                    active_embeddings[modality].weight,
                )
            )

    def test_graph_ttnc_transfer_reaches_active_composable_embedding(self):
        cfg = build_config(
            use_level1=True,
            use_composable_level1=True,
            use_graph_pretrained_code_embeddings=True,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_path = Path(tmp_dir) / "graph_embeddings.pt"
            ttnc_embeddings = torch.full(
                (cfg.ttnc_vocab_size, cfg.embedding_dim),
                3.5,
            )
            torch.save(
                {
                    "embedding_dim": cfg.embedding_dim,
                    "cpt_embeddings": torch.full(
                        (cfg.cpt_vocab_size, cfg.embedding_dim), 1.5
                    ),
                    "icd_embeddings": torch.full(
                        (cfg.icd_vocab_size, cfg.embedding_dim), 2.5
                    ),
                    "ttnc_embeddings": ttnc_embeddings,
                },
                embedding_path,
            )
            cfg.graph_embedding_path = str(embedding_path)
            model = HierarchicalClaimsModel(cfg)

        self.assertTrue(
            torch.allclose(
                model.context_level1_composer.ttnc_embedding.weight,
                ttnc_embeddings,
            )
        )

    def test_wristband_regularizer_is_calibrated_finite_and_differentiable(self):
        regularizer = WristbandGaussianRegularizer(
            embedding_dim=4,
            sample_size=8,
            calibration_reps=4,
        )
        embeddings = torch.randn(12, 4, requires_grad=True)
        loss, diagnostics = regularizer(embeddings)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(set(diagnostics), {"repulsion", "radial", "moment"})
        loss.backward()
        self.assertTrue(torch.isfinite(embeddings.grad).all())

    def test_vicreg_level2_baseline_loss_positive(self):
        cfg = build_config(ssl_objective_type="vicreg", target_encoder_mode="ema")
        model = HierarchicalClaimsModel(cfg)
        batch = make_batch(cfg)

        outputs = model.training_forward(*batch)
        self.assertGreater(outputs["ssl_loss_lvl2"].item(), 0)
        self.assertGreater(outputs["vicreg_loss_lvl2"].item(), 0)

    def test_sigreg_level2_loss_is_finite_and_backpropagates(self):
        cfg = build_config(ssl_objective_type="sigreg", target_encoder_mode="shared")
        model = HierarchicalClaimsModel(cfg)
        batch = make_batch(cfg)

        outputs = model.training_forward(*batch)
        self.assertTrue(torch.isfinite(outputs["ssl_loss_lvl2"]).all().item())

        outputs["loss"].backward()
        grad = next(
            param.grad
            for param in model.context_encoder_lvl2.parameters()
            if param.grad is not None
        )
        self.assertTrue(torch.isfinite(grad).all())

    def test_lejepa_sigreg_uses_convex_lambda_mixing(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            sigreg_formulation="lejepa_convex",
            sigreg_weight_lvl2=0.02,
        )
        objective = SIGRegObjective(cfg)
        prediction = torch.randn(12, cfg.embedding_dim)
        target = torch.randn(12, cfg.embedding_dim)

        metrics = objective.compute(prediction, target, level="2")

        expected = (
            metrics["predictive"] * 0.98
            + metrics["diagnostics"]["sigreg_raw"] * 0.02
        )
        self.assertTrue(torch.allclose(metrics["total"], expected))
        self.assertTrue(torch.isfinite(metrics["total"]))

    def test_lejepa_sigreg_statistic_scales_with_sample_count(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            sigreg_formulation="lejepa_convex",
        )
        objective = SIGRegObjective(cfg)
        embeddings = torch.randn(10, cfg.embedding_dim)

        torch.manual_seed(123)
        original = objective._characteristic_function_distance(embeddings)
        torch.manual_seed(123)
        duplicated = objective._characteristic_function_distance(
            embeddings.repeat(2, 1)
        )

        self.assertTrue(torch.allclose(duplicated, original * 2.0, rtol=1e-5))

    def test_logvars_can_be_frozen_after_configured_epoch(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            freeze_logvars_after_epoch=0,
        )
        model = HierarchicalClaimsModel(cfg)

        self.assertTrue(
            all(param.requires_grad for param in model.log_vars.parameters())
        )
        model.on_train_epoch_start()

        self.assertTrue(model._logvars_frozen)
        self.assertTrue(
            all(not param.requires_grad for param in model.log_vars.parameters())
        )

    def test_task_loss_weight_scales_predictor_contribution_without_changing_raw_loss(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_predictor_head=True,
            task_loss_weight=0.25,
        )
        model = HierarchicalClaimsModel(cfg)
        zero = torch.tensor(0.0)
        raw_task_loss = torch.tensor(4.0)

        total_loss, returned_task_loss = model.calculate_total_loss(
            zero,
            zero,
            raw_task_loss,
            lvl2_weight=0.0,
            token_pred_loss=zero,
        )

        self.assertAlmostEqual(returned_task_loss.item(), 4.0, places=6)
        self.assertAlmostEqual(total_loss.item(), 1.0, places=6)

    def test_uncertainty_weight_does_not_cancel_for_single_objective(self):
        cfg = build_config(ssl_objective_type="sigreg", target_encoder_mode="shared")
        model = HierarchicalClaimsModel(cfg)
        zero = torch.tensor(0.0)
        raw_ssl_loss = torch.tensor(4.0)

        total_loss, _ = model.calculate_total_loss(
            zero,
            raw_ssl_loss,
            zero,
            lvl2_weight=1.0,
            token_pred_loss=zero,
        )
        total_loss.backward()

        self.assertAlmostEqual(total_loss.item(), 4.0, places=6)
        self.assertAlmostEqual(model.log_vars["ssl_lvl2"].grad.item(), -3.0, places=6)

    def test_default_patient_representation_is_pre_sae(self):
        cfg = Config()
        self.assertFalse(cfg.use_gated_fusion)

        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_sparse_autoencoder=True,
        )
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))

        self.assertTrue(
            torch.equal(
                outputs["patient_representation"],
                outputs["patient_representation_pre_sae"],
            )
        )

    def test_predictor_head_source_uses_configured_representation_dim(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_predictor_head=True,
            predictor_head_source="patient_representation_pre_sae",
            use_sparse_autoencoder=True,
            use_gated_fusion=True,
            use_context_pooled_patient_representation=True,
        )
        model = HierarchicalClaimsModel(cfg)
        batch = make_batch(cfg)

        outputs = model.training_forward(*batch)

        self.assertEqual(
            model.non_linear_predictor[0].in_features,
            cfg.patient_representation_dim,
        )
        self.assertEqual(
            outputs["predictor_head_input"].shape[-1],
            cfg.patient_representation_dim,
        )
        self.assertTrue(torch.isfinite(outputs["task_loss"]).all().item())

    def test_level2_prediction_block_dense_mode_off_preserves_shape(self):
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=5,
            use_level2_dense_prediction=False,
        )
        context_embeddings = torch.randn(2, 4, 8)
        ttnc_tokens = torch.randint(1, 8, (2, 4))

        patient_representation, prediction = block(context_embeddings, ttnc_tokens)
        self.assertEqual(patient_representation.shape, (2, 8))
        self.assertEqual(prediction.shape, (2, 4))

    def test_gru_short_left_padded_sequences_remain_input_sensitive(self):
        torch.manual_seed(0)
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=50,
            num_layers=1,
            dropout=0.0,
            rnn_type="gru",
        )
        block.eval()

        context_embeddings = torch.zeros(2, 50, 8)
        context_embeddings[0, -10:] = torch.randn(10, 8)
        context_embeddings[1, -10:] = torch.randn(10, 8) * 7.0
        ttnc_tokens = torch.zeros(2, 50, dtype=torch.long)
        ttnc_tokens[:, -10:] = torch.randint(1, 8, (2, 10))

        patient_representation, prediction, aux = block(
            context_embeddings,
            ttnc_tokens,
            return_aux=True,
        )

        self.assertGreater(patient_representation.norm(dim=1).min().item(), 0.0)
        self.assertFalse(torch.allclose(patient_representation[0], patient_representation[1]))
        self.assertFalse(torch.allclose(prediction[0], prediction[1]))
        self.assertEqual(aux["sequence_output"][:, :-10].count_nonzero().item(), 0)

    def test_gru_short_sequence_representation_depends_on_latest_claim(self):
        torch.manual_seed(1)
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=50,
            num_layers=1,
            dropout=0.0,
            rnn_type="gru",
        )
        block.eval()

        context_embeddings = torch.zeros(2, 50, 8)
        shared_history = torch.randn(9, 8)
        context_embeddings[:, -10:-1] = shared_history
        context_embeddings[0, -1] = torch.arange(1, 9, dtype=torch.float32)
        context_embeddings[1, -1] = torch.tensor(
            [8.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0, 4.0]
        )
        ttnc_tokens = torch.zeros(2, 50, dtype=torch.long)
        ttnc_tokens[:, -10:] = 1

        patient_representation, prediction = block(context_embeddings, ttnc_tokens)

        self.assertFalse(torch.allclose(patient_representation[0], patient_representation[1]))
        self.assertFalse(torch.allclose(prediction[0], prediction[1]))

    def test_level2_prediction_block_dense_mode_on_returns_last_k_plus_next(self):
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=5,
            use_level2_dense_prediction=True,
            observed_claim_k=2,
        )
        context_embeddings = torch.randn(2, 4, 8)
        ttnc_tokens = torch.randint(1, 8, (2, 4))

        patient_representation, prediction = block(context_embeddings, ttnc_tokens)
        self.assertEqual(patient_representation.shape, (2, 8))
        self.assertEqual(prediction.shape, (2, 3, 4))

    def test_level2_prediction_block_dense_mode_can_return_future_horizons(self):
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=5,
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            future_claim_k=2,
        )
        context_embeddings = torch.randn(2, 4, 8)
        ttnc_tokens = torch.randint(1, 8, (2, 4))

        patient_representation, prediction = block(context_embeddings, ttnc_tokens)
        self.assertEqual(patient_representation.shape, (2, 8))
        self.assertEqual(prediction.shape, (2, 5, 4))

    def test_level2_prediction_block_dense_bottleneck_preserves_shape(self):
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=5,
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            use_dense_decoder_bottleneck=True,
            dense_decoder_bottleneck_dim=5,
        )
        context_embeddings = torch.randn(2, 4, 8)
        ttnc_tokens = torch.randint(1, 8, (2, 4))

        patient_representation, prediction, aux_outputs = block(
            context_embeddings,
            ttnc_tokens,
            return_aux=True,
        )
        self.assertEqual(patient_representation.shape, (2, 8))
        self.assertEqual(prediction.shape, (2, 3, 4))
        self.assertEqual(aux_outputs["dense_decoder_latent"].shape, (2, 5))

    def test_level2_prediction_block_bifurcated_state_exposes_predictive_state(self):
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=5,
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            use_bifurcated_patient_state=True,
        )
        context_embeddings = torch.randn(2, 4, 8)
        ttnc_tokens = torch.randint(1, 8, (2, 4))

        patient_representation, prediction, aux_outputs = block(
            context_embeddings,
            ttnc_tokens,
            return_aux=True,
        )
        self.assertEqual(patient_representation.shape, (2, 8))
        self.assertEqual(prediction.shape, (2, 3, 4))
        self.assertEqual(aux_outputs["predictive_state"].shape, (2, 16))
        self.assertFalse(
            torch.allclose(aux_outputs["predictive_state"], aux_outputs["context_pooled"])
        )

    def test_level2_prediction_block_mean_residual_broadcasts_into_patient_path(self):
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=5,
            dropout=0.0,
            use_context_pooled_patient_representation=True,
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            use_bifurcated_patient_state=True,
            use_patient_state_mean_residual=True,
            patient_state_mean_residual_weight=0.5,
        )
        context_embeddings = torch.randn(2, 4, 8)
        ttnc_tokens = torch.randint(1, 8, (2, 4))

        patient_representation, prediction, aux_outputs = block(
            context_embeddings,
            ttnc_tokens,
            return_aux=True,
        )

        self.assertEqual(patient_representation.shape, (2, 16))
        self.assertEqual(prediction.shape, (2, 3, 4))
        self.assertEqual(aux_outputs["patient_mean_residual"].shape, (2, 16))
        self.assertFalse(
            torch.allclose(patient_representation, aux_outputs["context_pooled"])
        )

    def test_level2_prediction_block_predictive_bottleneck_narrows_predictive_state_only(self):
        block = Level2PredictionBlock(
            embed_dim=8,
            output_dim=4,
            cpt_vocab_size=16,
            icd_vocab_size=16,
            ttnc_vocab_size=8,
            max_seq_length=5,
            use_context_pooled_patient_representation=True,
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            use_bifurcated_patient_state=True,
            use_predictive_state_bottleneck=True,
            predictive_state_bottleneck_dim=5,
        )
        context_embeddings = torch.randn(2, 4, 8)
        ttnc_tokens = torch.randint(1, 8, (2, 4))

        patient_representation, prediction, aux_outputs = block(
            context_embeddings,
            ttnc_tokens,
            return_aux=True,
        )

        self.assertEqual(patient_representation.shape, (2, 16))
        self.assertEqual(prediction.shape, (2, 3, 4))
        self.assertEqual(aux_outputs["predictive_state_base"].shape, (2, 16))
        self.assertEqual(aux_outputs["predictive_state"].shape, (2, 5))

    def test_dense_target_builder_handles_left_padding_and_short_contexts(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
        )
        model = HierarchicalClaimsModel(cfg)

        cpt_tensor = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [4, 4, 4], [5, 5, 5]],
                [[0, 0, 0], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ],
            dtype=torch.long,
        )
        icd_tensor = torch.tensor(
            [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0], [4, 4, 4], [5, 5, 5]],
                [[0, 0, 0], [6, 6, 6], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ],
            dtype=torch.long,
        )
        ttnc_tensor = torch.tensor(
            [
                [0, 0, 0, 4, 5],
                [0, 6, 7, 8, 9],
            ],
            dtype=torch.long,
        )

        targets = model._build_level2_ssl_targets(cpt_tensor, icd_tensor, ttnc_tensor)
        self.assertTrue(
            torch.equal(
                targets["target_mask"],
                torch.tensor([[False, True, True], [True, True, True]]),
            )
        )

        expected_cpt = torch.tensor(
            [
                [[0, 0, 0], [4, 4, 4], [5, 5, 5]],
                [[7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ],
            dtype=torch.long,
        )
        expected_icd = expected_cpt.clone()
        expected_ttnc = torch.tensor(
            [
                [0, 4, 5],
                [7, 8, 9],
            ],
            dtype=torch.long,
        )

        expected_repr = model._get_ssl_target_encoder_lvl2()(
            expected_cpt,
            expected_icd,
            expected_ttnc,
        )
        self.assertTrue(torch.allclose(targets["target_repr"], expected_repr))

    def test_dense_target_builder_can_add_future_horizons_with_decay(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            observed_claim_loss_weight=0.0,
            future_claim_k=2,
            future_claim_loss_weight=0.5,
            future_claim_loss_decay=0.5,
            next_claim_loss_weight=1.0,
        )
        model = HierarchicalClaimsModel(cfg)

        cpt_tensor = torch.tensor(
            [
                [[0, 0, 0], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[0, 0, 0], [0, 0, 0], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ],
            dtype=torch.long,
        )
        icd_tensor = cpt_tensor.clone()
        ttnc_tensor = torch.tensor(
            [
                [0, 2, 3, 4, 5],
                [0, 0, 7, 8, 9],
            ],
            dtype=torch.long,
        )

        targets = model._build_level2_ssl_targets(cpt_tensor, icd_tensor, ttnc_tensor)

        expected_mask = torch.tensor(
            [
                [False, True, True, True, True],
                [False, True, True, True, False],
            ]
        )
        self.assertTrue(torch.equal(targets["target_mask"], expected_mask))

        expected_cpt = torch.tensor(
            [
                [[0, 0, 0], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[0, 0, 0], [7, 7, 7], [8, 8, 8], [9, 9, 9], [0, 0, 0]],
            ],
            dtype=torch.long,
        )
        expected_icd = expected_cpt.clone()
        expected_ttnc = torch.tensor(
            [
                [0, 2, 3, 4, 5],
                [0, 7, 8, 9, 0],
            ],
            dtype=torch.long,
        )
        expected_repr = model._get_ssl_target_encoder_lvl2()(
            expected_cpt,
            expected_icd,
            expected_ttnc,
        )
        self.assertTrue(torch.allclose(targets["target_repr"], expected_repr))
        self.assertTrue(
            torch.equal(
                targets["target_ttnc"].squeeze(1),
                torch.tensor([3, 8], dtype=torch.long),
            )
        )
        self.assertTrue(
            torch.allclose(
                targets["slot_weights"],
                torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 0.5, 0.25],
                        [0.0, 0.0, 1.0, 0.5, 0.0],
                    ]
                ),
            )
        )

    def test_dense_target_builder_supports_zero_observed_claims(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_level2_dense_prediction=True,
            observed_claim_k=0,
            observed_claim_loss_weight=0.0,
            next_claim_loss_weight=1.0,
        )
        model = HierarchicalClaimsModel(cfg)

        cpt_tensor = torch.tensor(
            [
                [[0, 0, 0], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]],
                [[0, 0, 0], [0, 0, 0], [7, 7, 7], [8, 8, 8], [9, 9, 9]],
            ],
            dtype=torch.long,
        )
        icd_tensor = cpt_tensor.clone()
        ttnc_tensor = torch.tensor(
            [
                [0, 2, 3, 4, 5],
                [0, 0, 7, 8, 9],
            ],
            dtype=torch.long,
        )

        targets = model._build_level2_ssl_targets(cpt_tensor, icd_tensor, ttnc_tensor)

        self.assertEqual(targets["target_repr"].shape[1], 1)
        self.assertTrue(torch.equal(targets["target_mask"], torch.tensor([[True], [True]])))
        self.assertTrue(torch.equal(targets["slot_weights"], torch.tensor([[1.0], [1.0]])))

        expected_cpt = torch.tensor(
            [
                [[5, 5, 5]],
                [[9, 9, 9]],
            ],
            dtype=torch.long,
        )
        expected_icd = expected_cpt.clone()
        expected_ttnc = torch.tensor(
            [
                [5],
                [9],
            ],
            dtype=torch.long,
        )
        expected_repr = model._get_ssl_target_encoder_lvl2()(
            expected_cpt,
            expected_icd,
            expected_ttnc,
        )
        self.assertTrue(torch.allclose(targets["target_repr"], expected_repr))

    def test_training_forward_supports_zero_observed_claims(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="ema",
            use_level2_dense_prediction=True,
            observed_claim_k=0,
            observed_claim_loss_weight=0.0,
            use_masked_next_claim_token_grounding=True,
            masked_next_claim_mask_ratio=0.7,
            sigreg_weight_lvl2=0.0,
        )
        model = HierarchicalClaimsModel(cfg)
        batch = make_batch(cfg)

        outputs = model.training_forward(*batch)

        self.assertTrue(torch.isfinite(outputs["ssl_loss_lvl2"]).all().item())
        self.assertTrue(torch.isfinite(outputs["loss"]).all().item())

    def test_clean_ssl_mode_disables_token_and_diffusion_heads_but_keeps_sae(self):
        cfg = build_config(
            clean_ssl_mode=True,
            use_sparse_autoencoder=True,
            use_gated_fusion=False,
            use_diffusion=True,
            use_token_prediction_head=True,
        )
        model = HierarchicalClaimsModel(cfg)
        batch = make_batch(cfg)

        outputs = model.training_forward(*batch)
        self.assertFalse(model.use_diffusion)
        self.assertFalse(model.use_token_prediction_head)
        self.assertTrue(model.use_sparse_autoencoder)
        self.assertGreaterEqual(outputs["sae_loss"].item(), 0.0)
        self.assertEqual(outputs["diffusion_loss"].item(), 0.0)

    def test_masked_next_claim_grounding_produces_finite_token_loss(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="hybrid",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            use_masked_next_claim_token_grounding=True,
            masked_next_claim_mask_ratio=0.5,
            masked_next_claim_token_weight=0.2,
            clean_ssl_mode=True,
        )
        model = HierarchicalClaimsModel(cfg)
        batch = make_batch(cfg)

        torch.manual_seed(0)
        outputs = model.training_forward(*batch)

        self.assertGreater(outputs["masked_next_claim_token_loss"].item(), 0.0)
        self.assertTrue(torch.isfinite(outputs["masked_next_claim_token_loss"]).all().item())
        self.assertEqual(
            outputs["masked_next_claim_cpt_logits"].shape,
            (2, cfg.max_cpt_tokens, cfg.cpt_vocab_size),
        )
        self.assertEqual(
            outputs["masked_next_claim_icd_logits"].shape,
            (2, cfg.max_icd_tokens, cfg.icd_vocab_size),
        )
        self.assertEqual(
            outputs["masked_target_cpt_tokens"].shape,
            (2, cfg.max_cpt_tokens),
        )
        self.assertEqual(
            outputs["masked_target_icd_tokens"].shape,
            (2, cfg.max_icd_tokens),
        )

    def test_masked_next_claim_grounding_can_disable_one_code_stream(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="hybrid",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            use_masked_next_claim_token_grounding=True,
            masked_next_claim_mask_ratio=0.5,
            masked_next_claim_use_cpt=False,
            masked_next_claim_use_icd=True,
            clean_ssl_mode=True,
        )
        model = HierarchicalClaimsModel(cfg)
        batch = make_batch(cfg)

        torch.manual_seed(0)
        outputs = model.training_forward(*batch)

        self.assertEqual(outputs["masked_next_claim_cpt_loss"].item(), 0.0)
        self.assertGreater(outputs["masked_next_claim_icd_loss"].item(), 0.0)
        self.assertGreater(outputs["masked_next_claim_token_loss"].item(), 0.0)

    def test_target_encoder_updates_only_in_ema_mode(self):
        ema_cfg = build_config(target_encoder_mode="ema")
        ema_model = HierarchicalClaimsModel(ema_cfg)
        ema_context = next(ema_model.context_encoder_lvl2.parameters())
        ema_target = next(ema_model.target_encoder_lvl2.parameters())
        ema_before = ema_target.detach().clone()
        with torch.no_grad():
            ema_context.add_(1.0)
        ema_model.update_target_encoders()
        self.assertFalse(torch.allclose(ema_before, ema_target))

        shared_cfg = build_config(target_encoder_mode="shared")
        shared_model = HierarchicalClaimsModel(shared_cfg)
        shared_context = next(shared_model.context_encoder_lvl2.parameters())
        shared_target = next(shared_model.target_encoder_lvl2.parameters())
        shared_before = shared_target.detach().clone()
        with torch.no_grad():
            shared_context.add_(1.0)
        shared_model.update_target_encoders()
        self.assertTrue(torch.allclose(shared_before, shared_target))

    def test_configure_optimizers_supports_adam_without_scheduler(self):
        cfg = build_config(
            optimizer_type="adam",
            weight_decay=0.0,
            scheduler_type="none",
        )
        model = HierarchicalClaimsModel(cfg)

        optimizers = model.configure_optimizers()

        self.assertEqual(len(optimizers), 1)
        optimizer = optimizers[0]
        self.assertEqual(optimizer.__class__.__name__, "Adam")
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(optimizer.param_groups[1]["weight_decay"], 0.0)

    def test_configure_optimizers_supports_cosine_warmup_scheduler(self):
        cfg = build_config(
            optimizer_type="adamw",
            scheduler_type="cosine",
            scheduler_warmup_epochs=1,
            scheduler_t_max=3,
            scheduler_eta_min=1e-6,
        )
        model = HierarchicalClaimsModel(cfg)

        optimizers, schedulers = model.configure_optimizers()

        self.assertEqual(len(optimizers), 1)
        self.assertEqual(optimizers[0].__class__.__name__, "AdamW")
        self.assertEqual(len(schedulers), 1)
        self.assertEqual(schedulers[0].__class__.__name__, "SequentialLR")

    def test_target_encoder_hybrid_updates_only_configured_level(self):
        hybrid_cfg = build_config(
            use_level1=True,
            target_encoder_mode="hybrid",
            target_encoder_mode_lvl1="shared",
            target_encoder_mode_lvl2="ema",
        )
        hybrid_model = HierarchicalClaimsModel(hybrid_cfg)

        lvl1_context = next(hybrid_model.context_encoder_lvl1.parameters())
        lvl1_target = next(hybrid_model.target_encoder_lvl1.parameters())
        lvl2_context = next(hybrid_model.context_encoder_lvl2.parameters())
        lvl2_target = next(hybrid_model.target_encoder_lvl2.parameters())

        lvl1_before = lvl1_target.detach().clone()
        lvl2_before = lvl2_target.detach().clone()

        with torch.no_grad():
            lvl1_context.add_(1.0)
            lvl2_context.add_(1.0)

        hybrid_model.update_target_encoders()

        self.assertIs(hybrid_model._get_ssl_target_encoder_lvl1(), hybrid_model.context_encoder_lvl1)
        self.assertIs(hybrid_model._get_ssl_target_encoder_lvl2(), hybrid_model.target_encoder_lvl2)
        self.assertTrue(torch.allclose(lvl1_before, lvl1_target))
        self.assertFalse(torch.allclose(lvl2_before, lvl2_target))

    def test_train_cv_probe_logs_improvement_vs_mean_target_baseline(self):
        cfg = build_config(mean_target_baseline_rmse=100.0)
        model = HierarchicalClaimsModel(cfg)
        model.repr_accumulator = [torch.randn(10, cfg.embedding_dim)]
        model.target_accumulator = [torch.linspace(1.0, 2.0, 10)]

        logged = {}

        def capture_log(name, value, *args, **kwargs):
            if torch.is_tensor(value):
                value = value.detach().cpu().item()
            logged[name] = value

        model.log = capture_log
        model.on_train_epoch_end()

        self.assertIn("train_epoch_cv_probe_rmse", logged)
        self.assertIn("mean_target_baseline_rmse", logged)
        self.assertIn("train_epoch_cv_probe_improvement_vs_mean_baseline", logged)
        self.assertIn("train_epoch_cv_probe_improvement_pct_vs_mean_baseline", logged)
        self.assertEqual(logged["mean_target_baseline_rmse"], 100.0)
        self.assertAlmostEqual(
            logged["train_epoch_cv_probe_improvement_vs_mean_baseline"],
            100.0 - logged["train_epoch_cv_probe_rmse"],
            places=5,
        )
        self.assertAlmostEqual(
            logged["train_epoch_cv_probe_improvement_pct_vs_mean_baseline"],
            100.0 * (100.0 - logged["train_epoch_cv_probe_rmse"]) / 100.0,
            places=5,
        )

    def test_smoke_vicreg_baseline_recipe(self):
        cfg = build_config(ssl_objective_type="vicreg", target_encoder_mode="ema")
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))
        self.assertTrue(torch.isfinite(outputs["loss"]).all().item())
        self.assertEqual(outputs["prediction_lvl2"].shape, (2, cfg.output_dim))

    def test_smoke_sigreg_core_recipe(self):
        cfg = build_config(ssl_objective_type="sigreg", target_encoder_mode="shared")
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))
        self.assertTrue(torch.isfinite(outputs["loss"]).all().item())
        self.assertEqual(outputs["prediction_lvl2"].shape, (2, cfg.output_dim))

    def test_smoke_sigreg_dense_recipe(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="shared",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
        )
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))
        self.assertTrue(torch.isfinite(outputs["loss"]).all().item())
        self.assertEqual(outputs["prediction_lvl2"].shape, (2, 3, cfg.output_dim))

    def test_smoke_sigreg_dense_multihorizon_recipe(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="hybrid",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            observed_claim_loss_weight=0.0,
            future_claim_k=2,
            future_claim_loss_weight=0.5,
            future_claim_loss_decay=0.5,
            use_masked_next_claim_token_grounding=True,
            masked_next_claim_mask_ratio=0.7,
        )
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))
        self.assertTrue(torch.isfinite(outputs["loss"]).all().item())
        self.assertEqual(outputs["prediction_lvl2"].shape, (2, 5, cfg.output_dim))
        self.assertGreaterEqual(outputs["dense_future_loss"].item(), 0.0)

    def test_world_model_dynamics_smoke_recipe(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="hybrid",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            observed_claim_loss_weight=0.0,
            future_claim_k=2,
            future_claim_loss_weight=0.0,
            use_masked_next_claim_token_grounding=True,
            masked_next_claim_mask_ratio=0.7,
            sigreg_weight_lvl2=0.0,
            use_world_model_dynamics=True,
            world_model_future_steps=3,
            world_model_summary_weight=0.5,
            clean_ssl_mode=True,
        )
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))

        self.assertTrue(torch.isfinite(outputs["loss"]).all().item())
        self.assertTrue(torch.isfinite(outputs["world_model_loss"]).all().item())
        self.assertTrue(torch.isfinite(outputs["world_model_next_loss"]).all().item())
        self.assertTrue(torch.isfinite(outputs["world_model_summary_loss"]).all().item())
        self.assertEqual(
            outputs["world_model_rollout"].shape,
            (2, cfg.world_model_future_steps, cfg.output_dim),
        )
        self.assertEqual(
            outputs["world_model_state"].shape,
            (2, cfg.world_model_state_dim),
        )
        self.assertGreaterEqual(outputs["world_model_loss"].item(), 0.0)

    def test_temporal_contrastive_smoke_recipe(self):
        cfg = build_config(
            ssl_objective_type="sigreg",
            target_encoder_mode="hybrid",
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            observed_claim_loss_weight=0.0,
            future_claim_k=2,
            future_claim_loss_weight=0.0,
            future_claim_loss_decay=0.5,
            use_masked_next_claim_token_grounding=True,
            masked_next_claim_mask_ratio=0.7,
            sigreg_weight_lvl2=0.0,
            use_temporal_contrastive=True,
            temporal_ssl_mode="cpc",
            temporal_loss_weight=0.1,
            temporal_context_k=2,
            temporal_future_steps=3,
            clean_ssl_mode=True,
        )
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))

        self.assertTrue(torch.isfinite(outputs["loss"]).all().item())
        self.assertTrue(torch.isfinite(outputs["temporal_contrastive_loss"]).all().item())
        self.assertGreaterEqual(outputs["temporal_contrastive_loss"].item(), 0.0)
        self.assertEqual(outputs["prediction_lvl2"].shape, (2, 5, cfg.output_dim))
        self.assertEqual(outputs["temporal_context_view"].shape[-1], cfg.temporal_projection_dim)
        self.assertEqual(outputs["temporal_future_view"].shape[-1], cfg.temporal_projection_dim)

    def test_graph_pretrained_embeddings_are_loaded_into_encoders(self):
        cfg = build_config(
            use_level1=True,
            use_graph_pretrained_code_embeddings=True,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_path = Path(tmp_dir) / "graph_embeddings.pt"
            cpt_embeddings = torch.full((cfg.cpt_vocab_size, cfg.embedding_dim), 1.5)
            icd_embeddings = torch.full((cfg.icd_vocab_size, cfg.embedding_dim), 2.5)
            ttnc_embeddings = torch.full((cfg.ttnc_vocab_size, cfg.embedding_dim), 3.5)
            torch.save(
                {
                    "embedding_dim": cfg.embedding_dim,
                    "cpt_embeddings": cpt_embeddings,
                    "icd_embeddings": icd_embeddings,
                    "ttnc_embeddings": ttnc_embeddings,
                },
                embedding_path,
            )
            cfg.graph_embedding_path = str(embedding_path)
            model = HierarchicalClaimsModel(cfg)

        self.assertTrue(torch.allclose(model.context_encoder_lvl2.cpt_embedding.weight.detach(), cpt_embeddings))
        self.assertTrue(torch.allclose(model.context_encoder_lvl2.icd_embedding.weight.detach(), icd_embeddings))
        self.assertTrue(torch.allclose(model.context_encoder_lvl2.ttnc_embedding.weight.detach(), ttnc_embeddings))
        self.assertTrue(torch.allclose(model.target_encoder_lvl2.cpt_embedding.weight.detach(), cpt_embeddings))
        self.assertTrue(torch.allclose(model.context_encoder_lvl1.cpt_embedding.weight.detach(), cpt_embeddings))
        self.assertTrue(torch.allclose(model.prediction_block_lvl2.ttnc_embedding.weight.detach(), ttnc_embeddings))

    def test_level1_predictive_loss_can_be_disabled_without_disabling_composition(self):
        cfg = build_config(
            use_level1=True,
            use_composable_level1=True,
            target_encoder_mode="shared",
            level1_predictive_weight=0.0,
            level1_marginal_regularizer="none",
            sigreg_weight_lvl1=0.0,
            sigreg_weight_lvl2=0.05,
            ssl_objective_type="sigreg",
            sigreg_formulation="lejepa_convex",
        )
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))

        self.assertGreater(outputs["level1_predictive_objective_raw"].item(), 0)
        self.assertAlmostEqual(outputs["ssl_loss_lvl1"].item(), 0.0, places=6)
        outputs["loss"].backward()
        self.assertTrue(any(
            parameter.grad is not None
            for parameter in model.context_level1_composer.parameters()
        ))

    def test_claim_pooling_variants_are_permutation_invariant_and_finite(self):
        tokens = torch.tensor([[[1, 2, 3, 0]]])
        permuted = torch.tensor([[[3, 1, 2, 0]]])
        for pooling_type in ("moments", "query_attention", "self_attention"):
            encoder = Level1Encoder(
                cpt_vocab_size=8,
                icd_vocab_size=8,
                embedding_dim=4,
                pooling_type=pooling_type,
                num_heads=2,
            )
            first, _ = encoder(tokens, "cpt")
            second, _ = encoder(permuted, "cpt")
            self.assertTrue(torch.isfinite(first).all())
            self.assertTrue(torch.allclose(first, second, atol=1e-5))

    def test_ttnc_embedding_can_be_shared_and_ordinalized(self):
        cfg = build_config(
            use_level1=True,
            use_composable_level1=True,
            target_encoder_mode="shared",
            share_ttnc_embeddings=True,
            use_ttnc_ordinal_embedding=True,
        )
        cfg.ttnc_id_to_token = {
            0: "<PAD>", 1: "<UNK>", 2: "ttnc_01_day", 3: "ttnc_02_wk"
        }
        model = HierarchicalClaimsModel(cfg)

        self.assertIs(
            model.prediction_block_lvl2.ttnc_embedding,
            model.context_level1_composer.ttnc_embedding,
        )
        self.assertGreater(
            model.prediction_block_lvl2.ttnc_ordinal_values[3].item(),
            model.prediction_block_lvl2.ttnc_ordinal_values[2].item(),
        )
        optimizers, _ = model.configure_optimizers()
        parameter_ids = [
            id(parameter)
            for group in optimizers[0].param_groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(parameter_ids), len(set(parameter_ids)))

    def test_query_decoder_masked_claim_and_multi_hypothesis_are_finite(self):
        cfg = build_config(
            use_level1=True,
            use_composable_level1=True,
            target_encoder_mode="shared",
            ssl_objective_type="sigreg",
            sigreg_formulation="lejepa_convex",
            sigreg_weight_lvl2=0.05,
            use_level2_dense_prediction=True,
            observed_claim_k=2,
            observed_claim_loss_weight=0.0,
            level2_decoder_type="cross_attention",
            use_masked_claim_jepa=True,
            use_multi_hypothesis_future=True,
            multi_hypothesis_count=4,
        )
        model = HierarchicalClaimsModel(cfg)
        outputs = model.training_forward(*make_batch(cfg))

        self.assertEqual(outputs["prediction_lvl2"].shape, (2, 3, cfg.output_dim))
        self.assertEqual(outputs["masked_claim_prediction"].shape, (2, cfg.output_dim))
        self.assertEqual(
            outputs["multi_hypothesis_predictions"].shape,
            (2, 4, cfg.output_dim),
        )
        self.assertTrue(torch.isfinite(outputs["masked_claim_jepa_loss"]))
        self.assertTrue(torch.isfinite(outputs["multi_hypothesis_loss"]))
        outputs["loss"].backward()

        empty_context = torch.zeros(1, cfg.max_claims_len, cfg.embedding_dim)
        empty_ttnc = torch.zeros(1, cfg.max_claims_len, dtype=torch.long)
        patient, prediction = model.prediction_block_lvl2(empty_context, empty_ttnc)
        self.assertTrue(torch.isfinite(patient).all())
        self.assertTrue(torch.isfinite(prediction).all())
        optimizers, _ = model.configure_optimizers()
        optimized_ids = {
            id(parameter)
            for group in optimizers[0].param_groups
            for parameter in group["params"]
        }
        self.assertIn(
            id(next(model.masked_claim_query_head.parameters())), optimized_ids
        )
        self.assertIn(
            id(next(model.multi_hypothesis_future_head.parameters())), optimized_ids
        )

    def test_graph_pretrained_embeddings_can_be_mixed_and_stream_selected(self):
        cfg = build_config(
            use_level1=True,
            use_graph_pretrained_code_embeddings=True,
            graph_embedding_mix=0.25,
            graph_transfer_ttnc=False,
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            embedding_path = Path(tmp_dir) / "graph_embeddings.pt"
            cpt_embeddings = torch.full((cfg.cpt_vocab_size, cfg.embedding_dim), 2.0)
            icd_embeddings = torch.full((cfg.icd_vocab_size, cfg.embedding_dim), 4.0)
            ttnc_embeddings = torch.full((cfg.ttnc_vocab_size, cfg.embedding_dim), 6.0)
            torch.save(
                {
                    "embedding_dim": cfg.embedding_dim,
                    "cpt_embeddings": cpt_embeddings,
                    "icd_embeddings": icd_embeddings,
                    "ttnc_embeddings": ttnc_embeddings,
                },
                embedding_path,
            )
            cfg.graph_embedding_path = str(embedding_path)
            torch.manual_seed(0)
            expected_model = HierarchicalClaimsModel(build_config(use_level1=True))
            expected_cpt = expected_model.context_encoder_lvl2.cpt_embedding.weight.detach().clone()
            expected_icd = expected_model.context_encoder_lvl2.icd_embedding.weight.detach().clone()
            expected_ttnc = expected_model.context_encoder_lvl2.ttnc_embedding.weight.detach().clone()

            torch.manual_seed(0)
            model = HierarchicalClaimsModel(cfg)

        mixed_cpt = expected_cpt * 0.75 + cpt_embeddings * 0.25
        mixed_icd = expected_icd * 0.75 + icd_embeddings * 0.25
        self.assertTrue(torch.allclose(model.context_encoder_lvl2.cpt_embedding.weight.detach(), mixed_cpt))
        self.assertTrue(torch.allclose(model.context_encoder_lvl2.icd_embedding.weight.detach(), mixed_icd))
        self.assertTrue(torch.allclose(model.context_encoder_lvl2.ttnc_embedding.weight.detach(), expected_ttnc))


if __name__ == "__main__":
    unittest.main()
