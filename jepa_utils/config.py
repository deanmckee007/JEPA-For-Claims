# utils/config.py
from dataclasses import dataclass, field

@dataclass
class Config:
    data_path: str = 'C:/Users/tmcke/Desktop/claims_data/training_set.parquet'
    min_ttnc_tokens: int = 3    # Min number of claims per patient (use >= 3)
    min_valid_claims: int = min_ttnc_tokens - 1 # Clean this filtering up
    max_cpt_tokens: int = 5    # Max number of procedures per claim
    max_icd_tokens: int = 5  # Max number of diagnoses per claim
    max_claims_len: int = 50   # Max number of claims per patient
    embedding_dim: int = 128    
    hidden_dim: int = 200
    rnn_hidden_dim: int = 400
    output_dim: int = embedding_dim       # Dimension of the final representation
    num_layers: int = 2         # For the prediction transformer and RNNs
    num_heads: int = 4         # For the prediction transformer
    ff_hidden_dim: int = 200   # For the prediction transformer
    dropout: float = 0.00
    rnn_type: str = 'gru'       # Options: 'transformer', 'lstm', 'gru'
    lr: float = 7e-3            # Default, but overridden by LR finder
    adapter_lr: float = 1e-4    # LR for encoder adapter layers during Stage 2
    generator_lr: float = 5e-4  # LR for generator modules during Stage 2
    epochs: int = 25
    ema_decay: float = 0.999    # Higher value = less lagged updates to target encoder (use < 1)
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
    entropy_adjustment_factor = .2  # Use entropy to adjust sampling for each token type
    lambda_entropy = 0.05  # Adjust this value as needed
    max_generated_tokens = 10
    use_token_rarity = False  # Whether to use token rarity scores
    use_code_attention = False  # Whether to use attention in code pooling
    use_variance_embeddings = False  # Whether to include variance embeddings in aggregation
    use_aggregate_attention = False  # Whether to use attention pooling on aggregates
    use_component_attention = False  # Whether to use component-level attention pooling
    use_predictor_head = False
    use_grad_print = False
    use_na_targets = False
    use_level1 = False
    use_lr_find = False
    use_plotting = False
    use_zero_target_mask = True # Zeros are bad data in my DS - used with use_na_targets
    use_token_prediction_head = False
    use_generative_save = True
    use_sparse_autoencoder: bool = True
    use_gated_fusion: bool = True
    use_diffusion: bool = True  # Toggle diffusion-based claim generation
    # Whether to run a pretraining phase for the diffusion generator before
    # training the main hierarchical model. Kept ``True`` for backwards
    # compatibility.
    pretrain_diffusion: bool = False
    pretrain_diffusion_epochs: int = 0
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
    debug_low_threshold: bool = False
    debug_generation: bool = False
    out_encoder_ckpt: str = "encoder_only.ckpt"
    pretrained_encoder_ckpt: str = "encoder_only.ckpt"
    sae_hidden_dim: int = 256
    sae_k: int = 10
    gating_hidden_dim: int = 256
    # Multi-stage training epochs. If all set to 0, a single training stage is
    # executed as before.
    representation_pretrain_epochs: int = 100
    generator_train_epochs: int = 0
    joint_train_epochs: int = 0
    use_context_pooled_patient_representation: bool = True
    patient_representation_dim: int = field(init=False)

    def __post_init__(self):
        self.patient_representation_dim = self.embedding_dim * (
            2 if self.use_context_pooled_patient_representation else 1
        )
