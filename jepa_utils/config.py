# utils/config.py
from ast import literal_eval
from dataclasses import dataclass, field


TRAINING_RECIPES = {
    "custom": {},
    "vicreg_baseline": {
        "ssl_objective_type": "vicreg",
        "target_encoder_mode": "ema",
        "use_level2_dense_prediction": False,
        "clean_ssl_mode": True,
    },
    "sigreg_core": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "shared",
        "use_level2_dense_prediction": False,
        "clean_ssl_mode": True,
    },
    "sigreg_dense": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "shared",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
    },
    "sigreg_dense_hybrid": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
    },
    "sigreg_dense_hybrid_repr": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 20,
    },
    "sigreg_dense_hybrid_dollar": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 10,
    },
    "sigreg_dense_hybrid_dollar_masked_grounding": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 10,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.5,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
    },
    "sigreg_dense_hybrid_dollar_masked_grounding_sigreg0": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 10,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
        "sigreg_weight_lvl2": 0.0,
    },
    "composable_level1_none": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level1": True,
        "use_composable_level1": True,
        "level1_marginal_regularizer": "none",
        "sigreg_weight_lvl1": 0.0,
        "sigreg_weight_lvl2": 0.0,
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
    },
    "composable_level1_sigreg": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level1": True,
        "use_composable_level1": True,
        "level1_marginal_regularizer": "sigreg",
        "sigreg_weight_lvl1": 0.0,
        "sigreg_weight_lvl2": 0.0,
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
    },
    "composable_level1_lejepa": {
        "ssl_objective_type": "sigreg",
        "sigreg_formulation": "lejepa_convex",
        "target_encoder_mode": "shared",
        "use_level1": True,
        "use_composable_level1": True,
        "level1_marginal_regularizer": "sigreg",
        "sigreg_weight_lvl1": 0.0,
        "sigreg_weight_lvl2": 0.05,
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
    },
    "composable_level1_capi": {
        "ssl_objective_type": "sigreg",
        "sigreg_formulation": "lejepa_convex",
        "target_encoder_mode": "shared",
        "use_level1": True,
        "use_composable_level1": True,
        "level1_marginal_regularizer": "sigreg",
        "sigreg_weight_lvl1": 0.0,
        "sigreg_weight_lvl2": 0.05,
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
        "use_claim_prototypes": True,
        "claim_prototype_count": 64,
        "claim_prototype_weight": 0.05,
    },
    "composable_level1_wristband": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level1": True,
        "use_composable_level1": True,
        "level1_marginal_regularizer": "wristband",
        "sigreg_weight_lvl1": 0.0,
        "sigreg_weight_lvl2": 0.0,
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
    },
    "a9_world_model_claims": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "future_claim_k": 2,
        "future_claim_loss_weight": 0.0,
        "future_claim_loss_decay": 0.5,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 20,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
        "sigreg_weight_lvl2": 0.0,
        "use_world_model_dynamics": True,
        "world_model_future_steps": 3,
        "world_model_summary_weight": 0.5,
    },
    "a10_temporal_contrastive": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "future_claim_k": 2,
        "future_claim_loss_weight": 0.0,
        "future_claim_loss_decay": 0.5,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 24,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
        "sigreg_weight_lvl2": 0.0,
        "use_temporal_contrastive": True,
        "temporal_ssl_mode": "cpc",
        "temporal_loss_weight": 0.1,
        "temporal_context_k": 4,
        "temporal_future_steps": 3,
    },
    "a11_graph_ssl_transfer": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 24,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
        "sigreg_weight_lvl2": 0.0,
        "use_graph_pretrained_code_embeddings": True,
    },
    "sigreg_dense_hybrid_dollar_masked_grounding_multihorizon": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.0,
        "future_claim_k": 2,
        "future_claim_loss_weight": 0.5,
        "future_claim_loss_decay": 0.5,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 10,
        "use_masked_next_claim_token_grounding": True,
        "masked_next_claim_mask_ratio": 0.7,
        "masked_next_claim_token_weight": 0.2,
        "masked_next_claim_include_ttnc": False,
    },
    "sigreg_dense_hybrid_dollar_cost_head": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 10,
        "use_predictor_head": True,
        "task_loss_weight": 0.25,
    },
    "sigreg_dense_hybrid_dollar_bifurcated_state": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 10,
        "use_bifurcated_patient_state": True,
    },
    "sigreg_dense_hybrid_dollar_bifurcated_state_bottleneck128": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 10,
        "use_bifurcated_patient_state": True,
        "use_predictive_state_bottleneck": True,
        "predictive_state_bottleneck_dim": 128,
    },
    "sigreg_dense_hybrid_dollar_bifurcated_state_residual": {
        "ssl_objective_type": "sigreg",
        "target_encoder_mode": "hybrid",
        "use_level2_dense_prediction": True,
        "observed_claim_k": 2,
        "observed_claim_loss_weight": 0.25,
        "next_claim_loss_weight": 1.0,
        "clean_ssl_mode": True,
        "representation_pretrain_epochs": 10,
        "use_bifurcated_patient_state": True,
        "use_patient_state_mean_residual": True,
        "patient_state_mean_residual_weight": 0.5,
    },
}

# Preserve the historical composable recipe while giving the validated
# natural-missingness treatment an explicit, reusable identity.
TRAINING_RECIPES["composable_level1_sigreg_any_code"] = {
    **TRAINING_RECIPES["composable_level1_sigreg"],
    "claim_inclusion_policy": "any_code",
}

# Selected by the August 2026 core-architecture sweep.  A small explicit
# Level-1 predictive weight preserves CPT/ICD compositional alignment while
# leaving Level 2 free to consume the composed claim representation.
TRAINING_RECIPES["composable_level1_lejepa_any_code"] = {
    **TRAINING_RECIPES["composable_level1_lejepa"],
    "claim_inclusion_policy": "any_code",
    "level1_predictive_weight": 0.1,
}

# LeVJEPA claims port: learn patient-history invariance from one complete
# context and multiple independently thinned views.  The projector is used
# only by the pretraining loss; downstream consumers continue to read the
# canonical pre-SAE patient representation.
TRAINING_RECIPES["levjepa_patient_views"] = {
    **TRAINING_RECIPES["composable_level1_lejepa_any_code"],
    "sigreg_formulation": "levjepa_additive",
    "sigreg_weight_lvl2": 0.02,
    "sigreg_num_slices": 1024,
    "level1_marginal_regularizer": "none",
    "use_level2_dense_prediction": False,
    "use_masked_next_claim_token_grounding": False,
    "use_sparse_autoencoder": False,
    "use_levjepa_patient_views": True,
    "levjepa_num_local_views": 2,
    "levjepa_claim_drop_ratio": 0.3,
    "levjepa_projector_hidden_dim": 2048,
    "levjepa_projector_output_dim": 256,
    "use_eval_polyak_average": True,
    "eval_polyak_decay": 0.9999,
    "eval_polyak_update_interval": 32,
    "freeze_logvars_after_epoch": 0,
}

@dataclass
class Config:
    data_path: str = 'C:/Users/tmcke/OneDrive/Desktop/claims_data/training_set.parquet'
    data_contract_path: str | None = None
    create_data_contract_if_missing: bool = False
    train_split_fraction: float = 0.70
    val_split_fraction: float = 0.15
    test_split_fraction: float = 0.15
    evaluation_split: str = "val"
    vocab_min_freq: int = 5
    data_contract_hash: str | None = None
    vocab_hash: str | None = None
    allow_legacy_checkpoint_loading: bool = False
    train_recipe: str = "custom"
    # ``complete_only`` preserves the historical cohort. ``any_code`` retains
    # TTNC-delimited claims with at least one observed code family so the
    # composable encoder can learn natural missing-modality states.
    claim_inclusion_policy: str = "complete_only"
    # Optional evaluation-only view used for common-cohort comparisons. It is
    # deliberately excluded from checkpoint compatibility because it changes
    # neither training nor architecture.
    evaluation_claim_inclusion_policy: str | None = None
    # Training-only corruption applied to Level-2 context claims that contain
    # both modalities. Targets and naturally single-modality claims stay intact.
    context_cpt_dropout_prob: float = 0.0
    context_icd_dropout_prob: float = 0.0
    min_ttnc_tokens: int = 3    # Min number of claims per patient (use >= 3)
    min_valid_claims: int = min_ttnc_tokens - 1 # Clean this filtering up
    max_cpt_tokens: int = 5    # Max number of procedures per claim
    max_icd_tokens: int = 5  # Max number of diagnoses per claim
    max_claims_len: int = 50   # Max number of claims per patient
    embedding_dim: int = 128    
    hidden_dim: int = 200
    rnn_hidden_dim: int = 400
    output_dim: int | None = None         # Dimension of the final representation
    num_layers: int = 2         # For the prediction transformer and RNNs
    num_heads: int = 4         # For the prediction transformer
    ff_hidden_dim: int = 200   # For the prediction transformer
    dropout: float = 0.00
    rnn_type: str = 'gru'       # Options: 'transformer', 'lstm', 'gru'
    lr: float = 7e-3            # Default, but overridden by LR finder
    adapter_lr: float = 1e-4    # LR for encoder adapter layers during Stage 2
    generator_lr: float = 5e-4  # LR for generator modules during Stage 2
    optimizer_type: str = "adamw"
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    weight_decay: float = 1e-4
    scheduler_type: str = "step"
    scheduler_step_size: int = 2
    scheduler_gamma: float = 0.5
    scheduler_t_max: int | None = None
    scheduler_eta_min: float = 0.0
    scheduler_warmup_epochs: int = 0
    scheduler_warmup_start_factor: float = 0.2
    epochs: int = 25
    ema_decay: float = 0.999    # Higher value = less lagged updates to target encoder (use < 1)
    # Evaluation-only parameter averaging. Unlike target-encoder EMA, these
    # weights never participate in a training forward pass.
    use_eval_polyak_average: bool = False
    eval_polyak_decay: float = 0.9999
    eval_polyak_update_interval: int = 32
    epsilon: float = 1e-4  
    var_penalty_scale_lvl1: float = 1.0
    cov_penalty_scale_lvl1: float = 0.015 # .01 *Results for downstream task is very sensitive to this*
    var_penalty_scale_lvl2: float = 1.0  
    cov_penalty_scale_lvl2: float = 0.025 # .025 *Results for downstream task is very sensitive to this*   
    target_var_lvl1: float = .1    # Default 1.0
    target_var_lvl2: float = .2    # Default 1.0
    amplification_power: float = 1.0
    steps_per_epoch: int = 220  # Update based on DataLoader
    level_2_weight: float = 1.0
    ssl_objective_type: str = "vicreg"
    target_encoder_mode: str = "ema"
    target_encoder_mode_lvl1: str | None = None
    target_encoder_mode_lvl2: str | None = None
    clean_ssl_mode: bool = False
    entropy_adjustment_factor = .2  # Use entropy to adjust sampling for each token type
    lambda_entropy = 0.05  # Adjust this value as needed
    max_generated_tokens = 10
    use_token_rarity = False  # Whether to use token rarity scores
    use_code_attention = False  # Whether to use attention in code pooling
    use_variance_embeddings = False  # Whether to include variance embeddings in aggregation
    use_aggregate_attention = False  # Whether to use attention pooling on aggregates
    use_component_attention = False  # Whether to use component-level attention pooling
    use_predictor_head = False
    task_loss_weight: float = 1.0
    predictor_head_source: str = "context_mean_pool"
    use_grad_print = False
    use_na_targets = False
    use_level1 = False
    use_composable_level1: bool = False
    # Keep Level-1 representation flow independent from its bidirectional
    # CPT<->ICD predictive objective.  A zero weight leaves the composable
    # claim path active while removing that auxiliary loss.
    level1_predictive_weight: float = 1.0
    level1_marginal_regularizer: str = "none"
    level1_marginal_weight: float = 0.1
    level1_marginal_max_samples: int = 128
    claim_pooling_type: str = "moments"  # moments | query_attention | self_attention
    claim_pooling_num_heads: int = 4
    claim_pooling_use_rarity: bool = False
    wristband_calibration_reps: int = 16
    use_lr_find = False
    use_plotting = False
    use_zero_target_mask = True # Zeros are bad data in my DS - used with use_na_targets
    use_token_prediction_head = False
    allow_stage1_token_prediction_head: bool = False
    use_generative_save = True
    use_sparse_autoencoder: bool = True
    # The fused SAE readout is experimental. Keep the raw pre-SAE patient
    # representation canonical unless the fusion path has an explicit loss.
    use_gated_fusion: bool = False
    use_diffusion: bool = True  # Toggle diffusion-based claim generation
    use_level2_dense_prediction: bool = False
    level2_decoder_type: str = "dense"  # dense | cross_attention
    observed_claim_k: int = 2
    observed_claim_loss_weight: float = 0.25
    future_claim_k: int = 0
    future_claim_loss_weight: float = 0.5
    future_claim_loss_decay: float = 0.5
    next_claim_loss_weight: float = 1.0
    ttnc_in_composer: bool = True
    ttnc_in_sequence: bool = True
    share_ttnc_embeddings: bool = False
    use_ttnc_ordinal_embedding: bool = False
    use_masked_claim_jepa: bool = False
    masked_claim_jepa_weight: float = 0.1
    masked_claim_jepa_num_heads: int = 4
    use_multi_hypothesis_future: bool = False
    multi_hypothesis_count: int = 4
    multi_hypothesis_weight: float = 0.1
    multi_hypothesis_temperature: float = 0.1
    use_bifurcated_patient_state: bool = False
    use_predictive_state_bottleneck: bool = False
    predictive_state_bottleneck_dim: int | None = None
    use_patient_state_mean_residual: bool = False
    patient_state_mean_residual_weight: float = 0.5
    use_dense_decoder_bottleneck: bool = False
    dense_decoder_bottleneck_dim: int | None = None
    use_world_model_dynamics: bool = False
    world_model_state_dim: int | None = None
    world_model_future_steps: int | None = None
    world_model_next_weight: float = 1.0
    world_model_summary_weight: float = 0.0
    use_temporal_contrastive: bool = False
    temporal_ssl_mode: str = "cpc"
    temporal_loss_weight: float = 0.0
    temporal_projection_dim: int | None = None
    temporal_temperature: float = 0.1
    temporal_context_k: int = 4
    temporal_future_steps: int | None = None
    use_graph_pretrained_code_embeddings: bool = False
    graph_embedding_path: str | None = None
    graph_embedding_mix: float = 1.0
    graph_transfer_cpt: bool = True
    graph_transfer_icd: bool = True
    graph_transfer_ttnc: bool = True
    use_masked_next_claim_token_grounding: bool = False
    masked_next_claim_token_weight: float = 0.2
    masked_next_claim_mask_ratio: float = 0.5
    masked_next_claim_use_cpt: bool = True
    masked_next_claim_use_icd: bool = True
    masked_next_claim_include_ttnc: bool = False
    masked_next_claim_sort_target_tokens: bool = True
    masked_next_claim_hidden_dim: int | None = None
    # CAPI-style auxiliary over complete composed-claim targets.  This remains
    # opt-in so existing checkpoints and the canonical continuous
    # representation are unchanged.
    use_claim_prototypes: bool = False
    claim_prototype_count: int = 64
    claim_prototype_weight: float = 0.05
    claim_prototype_clustering_weight: float = 1.0
    claim_prototype_assignment_temperature: float = 0.06
    claim_prototype_clustering_temperature: float = 0.12
    claim_prototype_student_temperature: float = 0.12
    claim_prototype_sinkhorn_iterations: int = 3
    sigreg_weight_lvl1: float = 0.1
    sigreg_weight_lvl2: float = 0.1
    sigreg_num_slices: int = 256
    sigreg_num_points: int = 17
    sigreg_formulation: str = "legacy_additive"
    # LeVJEPA-style patient-history view objective. Local views thin valid
    # context claims independently while always retaining the most recent
    # claim. The projector is training-only: canonical downstream features do
    # not pass through it.
    use_levjepa_patient_views: bool = False
    levjepa_num_local_views: int = 2
    levjepa_claim_drop_ratio: float = 0.3
    levjepa_projector_hidden_dim: int = 2048
    levjepa_projector_output_dim: int = 256
    intermediate_sequence_supervision_weight: float = 0.0
    # Whether to run a pretraining phase for the diffusion generator before
    # training the main hierarchical model. Kept ``True`` for backwards
    # compatibility.
    pretrain_diffusion: bool = False
    pretrain_diffusion_epochs: int = 3
    diffusion_weight: float = 1.0  # Weight for diffusion loss when joint training
    diffusion_type: str = "discrete"  # continuous | discrete
    diffusion_steps: int = 100
    cpt_prob_agg: str = "max"  # max | mean | sum
    ttnc_temperature: float = 1.0
    base_cpt_threshold: float = 0.5
    base_icd_threshold: float = 0.5
    fine_tune_embeddings: bool = False
    sae_weight: float = 1.0
    freeze_transferred_embeddings: bool = False
    freeze_encoder_at_stage2: bool = True
    encoder_unfreeze_layers: int = 1
    current_stage: str = "stage1"
    seed: int = 42
    # The frozen patient split has its own identity. Leave this unset to retain
    # the historical behavior where ``seed`` controls both the split and model
    # randomness; set it explicitly when running multiple training seeds on one
    # immutable data contract.
    data_split_seed: int | None = None
    debug_low_threshold: bool = False
    debug_generation: bool = False
    out_encoder_ckpt: str = "encoder_only.ckpt"
    pretrained_encoder_ckpt: str = "encoder_only.ckpt"
    sae_hidden_dim: int = 256
    sae_k: int = 10
    gating_hidden_dim: int = 256
    # Multi-stage training epochs. If all set to 0, a single training stage is
    # executed as before.
    representation_pretrain_epochs: int = 20
    generator_train_epochs: int = 20
    joint_train_epochs: int = 0
    trainer_accelerator: str = "auto"
    trainer_devices: int = 1
    train_batch_size: int = 512
    eval_batch_size: int = 128
    checkpoint_save_top_k: int = 1
    checkpoint_save_last: bool = True
    checkpoint_every_n_epochs: int = 0
    checkpoint_dirpath: str = "checkpoints"
    checkpoint_monitor: str | None = None
    checkpoint_mode: str | None = None
    freeze_logvars_after_epoch: int | None = None
    mean_target_baseline_rmse: float | None = None
    use_context_pooled_patient_representation: bool = True
    patient_representation_dim: int = field(init=False)

    def __post_init__(self):
        self.refresh_derived_fields(sync_output_dim=True)

    def refresh_derived_fields(self, sync_output_dim: bool = False):
        if self.output_dim is None or sync_output_dim:
            self.output_dim = self.embedding_dim
        self.patient_representation_dim = self.embedding_dim * (
            2 if self.use_context_pooled_patient_representation else 1
        )


def apply_runtime_config_overrides(config: Config) -> Config:
    """Apply config presets that should consistently affect training/runtime."""
    config.refresh_derived_fields()
    config.ssl_objective_type = getattr(config, "ssl_objective_type", "vicreg").lower()
    config.sigreg_formulation = getattr(
        config,
        "sigreg_formulation",
        "legacy_additive",
    ).lower()
    config.target_encoder_mode = getattr(config, "target_encoder_mode", "ema").lower()
    config.optimizer_type = getattr(config, "optimizer_type", "adamw").lower()
    config.scheduler_type = getattr(config, "scheduler_type", "step").lower()
    config.evaluation_split = getattr(config, "evaluation_split", "val").lower()
    config.claim_inclusion_policy = getattr(
        config, "claim_inclusion_policy", "complete_only"
    ).lower()
    evaluation_policy = getattr(
        config, "evaluation_claim_inclusion_policy", None
    )
    config.evaluation_claim_inclusion_policy = (
        evaluation_policy.lower() if evaluation_policy is not None else None
    )
    if getattr(config, "target_encoder_mode_lvl1", None) is not None:
        config.target_encoder_mode_lvl1 = config.target_encoder_mode_lvl1.lower()
    if getattr(config, "target_encoder_mode_lvl2", None) is not None:
        config.target_encoder_mode_lvl2 = config.target_encoder_mode_lvl2.lower()

    if config.ssl_objective_type not in {"vicreg", "sigreg"}:
        raise ValueError(
            f"Unsupported ssl_objective_type={config.ssl_objective_type!r}. "
            "Expected 'vicreg' or 'sigreg'."
        )
    if config.sigreg_formulation not in {
        "legacy_additive",
        "lejepa_convex",
        "levjepa_additive",
    }:
        raise ValueError(
            "sigreg_formulation must be 'legacy_additive', 'lejepa_convex', "
            "or 'levjepa_additive'."
        )
    if config.sigreg_formulation == "lejepa_convex":
        for field_name in ("sigreg_weight_lvl1", "sigreg_weight_lvl2"):
            value = getattr(config, field_name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{field_name} must be in [0, 1] for lejepa_convex SIGReg."
                )
    elif config.sigreg_weight_lvl1 < 0 or config.sigreg_weight_lvl2 < 0:
        raise ValueError("Additive SIGReg weights must be non-negative.")

    if config.levjepa_num_local_views <= 0:
        raise ValueError("levjepa_num_local_views must be positive.")
    if not 0.0 <= config.levjepa_claim_drop_ratio < 1.0:
        raise ValueError("levjepa_claim_drop_ratio must be in the interval [0, 1).")
    if config.levjepa_projector_hidden_dim <= 0:
        raise ValueError("levjepa_projector_hidden_dim must be positive.")
    if config.levjepa_projector_output_dim <= 0:
        raise ValueError("levjepa_projector_output_dim must be positive.")
    if config.use_levjepa_patient_views:
        if config.ssl_objective_type != "sigreg":
            raise ValueError("LeVJEPA patient views require ssl_objective_type='sigreg'.")
        if config.sigreg_formulation != "levjepa_additive":
            raise ValueError(
                "LeVJEPA patient views require sigreg_formulation='levjepa_additive'."
            )
        if config.target_encoder_mode != "shared":
            raise ValueError("LeVJEPA patient views require a shared target encoder.")

    config.level1_marginal_regularizer = getattr(
        config, "level1_marginal_regularizer", "none"
    ).lower()
    if config.level1_marginal_regularizer not in {"none", "sigreg", "wristband"}:
        raise ValueError(
            "level1_marginal_regularizer must be 'none', 'sigreg', or 'wristband'."
        )
    if config.use_composable_level1 and not config.use_level1:
        raise ValueError("use_composable_level1=True requires use_level1=True.")
    config.claim_pooling_type = getattr(config, "claim_pooling_type", "moments").lower()
    if config.claim_pooling_type not in {"moments", "query_attention", "self_attention"}:
        raise ValueError(
            "claim_pooling_type must be 'moments', 'query_attention', or 'self_attention'."
        )
    if config.claim_pooling_num_heads <= 0 or config.embedding_dim % config.claim_pooling_num_heads:
        raise ValueError("claim_pooling_num_heads must evenly divide embedding_dim.")
    if config.level1_predictive_weight < 0:
        raise ValueError("level1_predictive_weight must be non-negative.")
    config.level2_decoder_type = getattr(config, "level2_decoder_type", "dense").lower()
    if config.level2_decoder_type not in {"dense", "cross_attention"}:
        raise ValueError("level2_decoder_type must be 'dense' or 'cross_attention'.")
    if config.masked_claim_jepa_weight < 0:
        raise ValueError("masked_claim_jepa_weight must be non-negative.")
    if config.masked_claim_jepa_num_heads <= 0 or config.embedding_dim % config.masked_claim_jepa_num_heads:
        raise ValueError("masked_claim_jepa_num_heads must evenly divide embedding_dim.")
    if config.multi_hypothesis_count < 2:
        raise ValueError("multi_hypothesis_count must be at least 2.")
    if config.multi_hypothesis_weight < 0 or config.multi_hypothesis_temperature <= 0:
        raise ValueError("multi-hypothesis weight must be non-negative and temperature positive.")

    if config.target_encoder_mode not in {"ema", "shared", "hybrid"}:
        raise ValueError(
            f"Unsupported target_encoder_mode={config.target_encoder_mode!r}. "
            "Expected 'ema', 'shared', or 'hybrid'."
        )

    if config.optimizer_type not in {"adamw", "adam"}:
        raise ValueError(
            f"Unsupported optimizer_type={config.optimizer_type!r}. "
            "Expected 'adamw' or 'adam'."
        )

    if config.scheduler_type not in {"step", "cosine", "none"}:
        raise ValueError(
            f"Unsupported scheduler_type={config.scheduler_type!r}. "
            "Expected 'step', 'cosine', or 'none'."
        )
    if not 0.0 <= config.eval_polyak_decay < 1.0:
        raise ValueError("eval_polyak_decay must be in the interval [0, 1).")
    if config.eval_polyak_update_interval <= 0:
        raise ValueError("eval_polyak_update_interval must be positive.")

    split_total = (
        config.train_split_fraction
        + config.val_split_fraction
        + config.test_split_fraction
    )
    if abs(split_total - 1.0) > 1e-8:
        raise ValueError("train/val/test split fractions must sum to 1.0.")
    if config.train_split_fraction <= 0:
        raise ValueError("train_split_fraction must be positive.")
    if config.val_split_fraction < 0 or config.test_split_fraction < 0:
        raise ValueError("val/test split fractions must be non-negative.")
    if config.evaluation_split not in {"val", "test"}:
        raise ValueError("evaluation_split must be either 'val' or 'test'.")
    if config.vocab_min_freq <= 0:
        raise ValueError("vocab_min_freq must be positive.")
    if config.claim_inclusion_policy not in {"complete_only", "any_code"}:
        raise ValueError(
            "claim_inclusion_policy must be either 'complete_only' or 'any_code'."
        )
    if config.evaluation_claim_inclusion_policy not in {
        None,
        "complete_only",
        "any_code",
    }:
        raise ValueError(
            "evaluation_claim_inclusion_policy must be None, 'complete_only', "
            "or 'any_code'."
        )
    for field_name in ("context_cpt_dropout_prob", "context_icd_dropout_prob"):
        probability = getattr(config, field_name)
        if not 0.0 <= probability < 1.0:
            raise ValueError(f"{field_name} must be in the interval [0, 1).")

    if config.target_encoder_mode == "hybrid":
        # Current stage-1 training is Level-2 dominated. The hybrid recipe keeps
        # optional Level 1 supervision on the shared encoder while using a
        # lagged EMA target for Level 2 future-claim supervision.
        if config.target_encoder_mode_lvl1 is None:
            config.target_encoder_mode_lvl1 = "shared"
        if config.target_encoder_mode_lvl2 is None:
            config.target_encoder_mode_lvl2 = "ema"
    else:
        if config.target_encoder_mode_lvl1 is None:
            config.target_encoder_mode_lvl1 = config.target_encoder_mode
        if config.target_encoder_mode_lvl2 is None:
            config.target_encoder_mode_lvl2 = config.target_encoder_mode

    for level_name, mode in (
        ("target_encoder_mode_lvl1", config.target_encoder_mode_lvl1),
        ("target_encoder_mode_lvl2", config.target_encoder_mode_lvl2),
    ):
        if mode not in {"ema", "shared"}:
            raise ValueError(
                f"Unsupported {level_name}={mode!r}. Expected 'ema' or 'shared'."
            )
    if config.use_levjepa_patient_views and (
        config.target_encoder_mode_lvl1 != "shared"
        or config.target_encoder_mode_lvl2 != "shared"
    ):
        raise ValueError(
            "LeVJEPA patient views require shared Level-1 and Level-2 targets."
        )

    if getattr(config, "clean_ssl_mode", False):
        config.use_token_prediction_head = False
        config.use_diffusion = False
        config.use_generative_save = False
        config.pretrain_diffusion = False

    if getattr(config, "use_masked_next_claim_token_grounding", False):
        if not (0.0 < config.masked_next_claim_mask_ratio <= 1.0):
            raise ValueError(
                "masked_next_claim_mask_ratio must be in the interval (0, 1]."
            )
        if config.masked_next_claim_token_weight < 0:
            raise ValueError(
                "masked_next_claim_token_weight must be non-negative."
            )
        if not any(
            (
                getattr(config, "masked_next_claim_use_cpt", True),
                getattr(config, "masked_next_claim_use_icd", True),
                getattr(config, "masked_next_claim_include_ttnc", False),
            )
        ):
            raise ValueError(
                "masked next-claim grounding must supervise at least one token stream."
            )
    if getattr(config, "use_claim_prototypes", False):
        if not config.use_composable_level1:
            raise ValueError(
                "use_claim_prototypes=True requires the composed Level-1 claim representation."
            )
        if config.claim_prototype_count < 2:
            raise ValueError("claim_prototype_count must be at least 2.")
        if config.claim_prototype_weight < 0:
            raise ValueError("claim_prototype_weight must be non-negative.")
        if config.claim_prototype_clustering_weight < 0:
            raise ValueError(
                "claim_prototype_clustering_weight must be non-negative."
            )
        for field_name in (
            "claim_prototype_assignment_temperature",
            "claim_prototype_clustering_temperature",
            "claim_prototype_student_temperature",
        ):
            if getattr(config, field_name) <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if config.claim_prototype_sinkhorn_iterations <= 0:
            raise ValueError(
                "claim_prototype_sinkhorn_iterations must be positive."
            )
    if config.observed_claim_k < 0:
        raise ValueError("observed_claim_k must be non-negative.")
    if config.future_claim_k < 0:
        raise ValueError("future_claim_k must be non-negative.")
    if getattr(config, "world_model_state_dim", None) is None:
        config.world_model_state_dim = config.patient_representation_dim
    if getattr(config, "world_model_future_steps", None) is None:
        config.world_model_future_steps = config.future_claim_k + 1
    if getattr(config, "temporal_projection_dim", None) is None:
        config.temporal_projection_dim = config.output_dim
    if getattr(config, "temporal_future_steps", None) is None:
        config.temporal_future_steps = config.future_claim_k + 1
    if config.observed_claim_loss_weight < 0:
        raise ValueError("observed_claim_loss_weight must be non-negative.")
    if config.next_claim_loss_weight < 0:
        raise ValueError("next_claim_loss_weight must be non-negative.")
    if config.future_claim_loss_weight < 0:
        raise ValueError("future_claim_loss_weight must be non-negative.")
    if config.future_claim_loss_decay < 0:
        raise ValueError("future_claim_loss_decay must be non-negative.")
    if config.world_model_future_steps <= 0:
        raise ValueError("world_model_future_steps must be positive.")
    if config.world_model_next_weight < 0:
        raise ValueError("world_model_next_weight must be non-negative.")
    if config.world_model_summary_weight < 0:
        raise ValueError("world_model_summary_weight must be non-negative.")
    config.temporal_ssl_mode = getattr(config, "temporal_ssl_mode", "cpc").lower()
    if config.temporal_ssl_mode not in {"cpc", "ts2vec"}:
        raise ValueError("temporal_ssl_mode must be either 'cpc' or 'ts2vec'.")
    if config.temporal_loss_weight < 0:
        raise ValueError("temporal_loss_weight must be non-negative.")
    if config.temporal_projection_dim <= 0:
        raise ValueError("temporal_projection_dim must be positive.")
    if config.temporal_temperature <= 0:
        raise ValueError("temporal_temperature must be positive.")
    if config.temporal_context_k <= 0:
        raise ValueError("temporal_context_k must be positive.")
    if config.temporal_future_steps <= 0:
        raise ValueError("temporal_future_steps must be positive.")
    if config.future_claim_k > 0 and not config.use_level2_dense_prediction:
        raise ValueError(
            "future_claim_k requires use_level2_dense_prediction=True."
        )
    if config.use_world_model_dynamics and not config.use_level2_dense_prediction:
        raise ValueError(
            "use_world_model_dynamics requires use_level2_dense_prediction=True."
        )
    if config.use_temporal_contrastive and config.temporal_future_steps > config.future_claim_k + 1:
        raise ValueError(
            "temporal_future_steps cannot exceed future_claim_k + 1 for the current target builder."
        )
    if getattr(config, "use_graph_pretrained_code_embeddings", False):
        graph_embedding_path = getattr(config, "graph_embedding_path", None)
        if graph_embedding_path is None or not str(graph_embedding_path).strip():
            raise ValueError(
                "graph_embedding_path must be provided when "
                "use_graph_pretrained_code_embeddings=True."
            )
        if not (0.0 <= config.graph_embedding_mix <= 1.0):
            raise ValueError("graph_embedding_mix must be in the interval [0, 1].")
        if not (
            config.graph_transfer_cpt
            or config.graph_transfer_icd
            or config.graph_transfer_ttnc
        ):
            raise ValueError(
                "At least one of graph_transfer_cpt, graph_transfer_icd, or "
                "graph_transfer_ttnc must be enabled when using graph-pretrained embeddings."
            )
    if getattr(config, "task_loss_weight", 1.0) < 0:
        raise ValueError("task_loss_weight must be non-negative.")
    if config.train_batch_size <= 0:
        raise ValueError("train_batch_size must be positive.")
    if config.eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive.")
    if config.checkpoint_save_top_k < 0:
        raise ValueError("checkpoint_save_top_k must be non-negative.")
    if config.checkpoint_every_n_epochs < 0:
        raise ValueError("checkpoint_every_n_epochs must be non-negative.")
    if not str(config.checkpoint_dirpath).strip():
        raise ValueError("checkpoint_dirpath must be a non-empty path.")
    if config.checkpoint_mode is not None:
        config.checkpoint_mode = config.checkpoint_mode.lower()
        if config.checkpoint_mode not in {"min", "max"}:
            raise ValueError("checkpoint_mode must be either 'min' or 'max'.")
    if config.freeze_logvars_after_epoch is not None and config.freeze_logvars_after_epoch < 0:
        raise ValueError("freeze_logvars_after_epoch must be non-negative when provided.")
    if config.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if not (0.0 < config.optimizer_beta1 < 1.0):
        raise ValueError("optimizer_beta1 must be in the interval (0, 1).")
    if not (0.0 < config.optimizer_beta2 < 1.0):
        raise ValueError("optimizer_beta2 must be in the interval (0, 1).")
    if config.scheduler_step_size <= 0:
        raise ValueError("scheduler_step_size must be positive.")
    if config.scheduler_gamma <= 0:
        raise ValueError("scheduler_gamma must be positive.")
    if config.scheduler_t_max is not None and config.scheduler_t_max <= 0:
        raise ValueError("scheduler_t_max must be positive when provided.")
    if config.scheduler_eta_min < 0:
        raise ValueError("scheduler_eta_min must be non-negative.")
    if config.scheduler_warmup_epochs < 0:
        raise ValueError("scheduler_warmup_epochs must be non-negative.")
    if not (0.0 < config.scheduler_warmup_start_factor <= 1.0):
        raise ValueError(
            "scheduler_warmup_start_factor must be in the interval (0, 1]."
        )
    config.predictor_head_source = getattr(
        config,
        "predictor_head_source",
        "context_mean_pool",
    ).lower()
    if config.predictor_head_source not in {
        "context_mean_pool",
        "context_max_pool",
        "context_pooled",
        "patient_representation_pre_sae",
        "patient_representation",
    }:
        raise ValueError(
            "predictor_head_source must be one of "
            "{'context_mean_pool', 'context_max_pool', 'context_pooled', "
            "'patient_representation_pre_sae', 'patient_representation'}."
        )

    return config


def _coerce_config_override(raw_value: str):
    lowered = raw_value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None

    try:
        return literal_eval(raw_value)
    except (ValueError, SyntaxError):
        return raw_value


def parse_config_overrides(raw_overrides: list[str] | None):
    parsed = {}
    for raw_override in raw_overrides or []:
        if "=" not in raw_override:
            raise ValueError(
                f"Invalid config override {raw_override!r}. Expected key=value."
            )
        key, raw_value = raw_override.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(
                f"Invalid config override {raw_override!r}. Override key cannot be empty."
            )
        parsed[key] = _coerce_config_override(raw_value.strip())
    return parsed


def apply_config_overrides(
    config: Config,
    raw_overrides: list[str] | None = None,
    overrides: dict | None = None,
) -> Config:
    merged_overrides = {}
    if overrides:
        merged_overrides.update(overrides)
    merged_overrides.update(parse_config_overrides(raw_overrides))

    for key, value in merged_overrides.items():
        if not hasattr(config, key):
            raise ValueError(
                f"Unknown config override {key!r}. Add it to Config before using --set."
            )
        setattr(config, key, value)

    config.refresh_derived_fields(
        sync_output_dim=(
            "embedding_dim" in merged_overrides and "output_dim" not in merged_overrides
        )
    )

    return config


def get_training_recipe_names():
    return sorted(TRAINING_RECIPES.keys())


def apply_training_recipe(config: Config, recipe_name: str | None) -> Config:
    recipe = (recipe_name or "custom").lower()
    if recipe not in TRAINING_RECIPES:
        raise ValueError(
            f"Unsupported train_recipe={recipe!r}. "
            f"Expected one of: {', '.join(get_training_recipe_names())}."
        )

    for key, value in TRAINING_RECIPES[recipe].items():
        setattr(config, key, value)

    config.refresh_derived_fields(
        sync_output_dim=(
            "embedding_dim" in TRAINING_RECIPES[recipe]
            and "output_dim" not in TRAINING_RECIPES[recipe]
        )
    )
    config.train_recipe = recipe
    return config
