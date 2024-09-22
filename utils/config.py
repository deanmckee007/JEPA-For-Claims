# utils/config.py
from dataclasses import dataclass

@dataclass
class Config:
    data_path: str = 'C:/Users/tmcke/Desktop/claims_data/training_set.parquet'
    min_ttnc_tokens: int = 3    # Min number of claims per patient (use >= 3)
    max_cpt_tokens: int = 5    # Max number of procedures per claim
    max_icd_tokens: int = 5  # Max number of diagnoses per claim
    max_claims_len: int = 50   # Max number of claims per patient
    embedding_dim: int = 400    
    hidden_dim: int = 400
    rnn_hidden_dim: int = 400
    output_dim: int = 400       # Dimension of the final representation
    num_layers: int = 4         # For the prediction transformer
    num_heads: int = 4          # For the prediction transformer
    ff_hidden_dim: int = 1024   # For the prediction transformer
    dropout: float = 0.1
    rnn_type: str = 'gru'       # Options: 'transformer', 'lstm', 'gru'
    lr: float = 1e-5            # Default, but overridden by LR finder
    ema_decay: float = 0.999    # Higher value = less lagged updates to target encoder (use < 1)
    epsilon: float = 1e-4  
    var_penalty_scale: float = 1.0     
    target_var: float = 0.3     # Default 1.0
    amplification_power: float = 1.0
    steps_per_epoch: int = 220  # Update based on DataLoader
    use_token_rarity = False  # Whether to use token rarity scores
    use_code_attention = True  # Whether to use attention in code pooling
    use_variance_embeddings = True  # Whether to include variance embeddings in aggregation
    use_aggregate_attention = True  # Whether to use attention pooling on aggregates
    use_component_attention = True  # Whether to use component-level attention pooling
    use_predictor_head = True
 
