# models/hierarchical_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import re
from jepa_models.encoders import ComposableClaimEncoder, Level1Encoder, Level2Encoder
from jepa_models.claim_prototypes import ClaimPrototypeObjective
from jepa_models.prediction_blocks import (
    Level1PredictionBlock,
    Level2PredictionBlock,
    LogitsGenerator,
    MaskedClaimTokenDecoder,
    MaskedClaimQueryHead,
    MultiHypothesisFutureHead,
)
from jepa_models.temporal_ssl import TemporalContrastiveHead
from jepa_models.world_model import ClaimsWorldModel
from jepa_models.diffusion import DiffusionModel
from jepa_models.discrete_diffusion import DiscreteDiffusionModel
from jepa_models.sparse_autoencoder import SparseAutoencoder
from jepa_models.ssl_objectives import (
    WristbandGaussianRegularizer,
    build_ssl_objective,
    sigreg_gaussian_distance,
)
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.metrics import calculate_rmse
from jepa_utils.tensor_utils import calculate_entropy
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


def _build_ttnc_ordinal_values(id_to_token, vocab_size):
    """Map TTNC bucket labels to normalized log-days while preserving unknowns."""
    values = torch.zeros(vocab_size, dtype=torch.float32)
    unit_days = {"day": 1.0, "wk": 7.0, "mo": 30.4375}
    for token_id, token in (id_to_token or {}).items():
        match = re.fullmatch(r"ttnc_(\d+)_(day|wk|mo)", str(token))
        if match and int(token_id) < vocab_size:
            values[int(token_id)] = math.log1p(
                int(match.group(1)) * unit_days[match.group(2)]
            )
    maximum = values.max()
    if maximum > 0:
        values = values / maximum
    return values

class HierarchicalClaimsModel(pl.LightningModule):
    """
    HierarchicalClaimsModel is a multi-level neural network for processing healthcare claims data. 
    It operates in two levels:
    - Level 1: Processes individual claims using CPT and ICD codes.
    - Level 2: Predicts the next claim representation using sequences of claims as context.
    It's a JEPA style architechture https://arxiv.org/abs/2301.08243, hopefully squeezing
    maximal 'juice' for a self supervised approach.
    It creates patient level representations by default and optionally does claims
    generation using the predicted representation as input.  For claims generation, an
    'anchor' procedure is sampled according to the probability distribution - encouraging
    exploration from claim to claim, and then transitioning to entropy modified thresholds for
    the remaining components on the claim to encourage within claim consistency/expectation.

    Key Components:
    ---------------
    - **Encoders:** 
        - Level1Encoder: Encodes individual claims using CPT OR ICD tokens.
        - Level2Encoder: Encodes a sequence of claims (CPT, ICD, TTNC tokens) with optional attention, token rarity, and variance embeddings.
    - **Prediction Blocks:** 
        - Level1PredictionBlock: Performs within-claim predictions of representations.
        - Level2PredictionBlock: Performs across-claims predictions of representations using an RNN or Transformer-based architecture.
    - **Logits Generator:** 
        - LogitsGenerator: Generates logits for CPT, ICD, and TTNC codes.

    Parameters:
    -----------
    config : dict
        Dictionary containing model configuration and hyperparameters:
        - cpt_vocab_size (int): Size of the CPT code vocabulary.
        - icd_vocab_size (int): Size of the ICD code vocabulary.
        - ttnc_vocab_size (int): Size of the TTNC code vocabulary.
        - embedding_dim (int): Dimensionality of embeddings for each token type.
        - max_claims_len (int): Maximum number of claims in a sequence.
        - dropout (float): Dropout rate used throughout the model.
        - rnn_type (str): Type of RNN for Level 2, e.g., 'lstm' or 'gru'. Can also be 'transformer'.
        - use_predictor_head (bool): Flag to include a regression prediction head.
        - use_token_prediction_head (bool): Flag to include a token prediction head in the model.
        - level_2_weight (float): Weight applied to the Level 2 loss component.
        - lr (float): Learning rate for optimization.
        - ema_decay (float): Decay factor for updating target encoders with EMA.
        - use_level1 (bool): Whether to use Level 1 encoding and prediction.

    Attributes:
    -----------
    - cpt_embedding (nn.Embedding): Embedding layer for CPT tokens.
    - icd_embedding (nn.Embedding): Embedding layer for ICD tokens.
    - ttnc_embedding (nn.Embedding): Embedding layer for TTNC tokens.
    - context_encoder_lvl1 (Level1Encoder): Encoder for the context (claims).
    - target_encoder_lvl1 (Level1Encoder): Encoder for target claims.
    - context_encoder_lvl2 (Level2Encoder): Encoder for sequences of claims as context.
    - target_encoder_lvl2 (Level2Encoder): Encoder for the target sequence (next claim).
    - prediction_block_lvl1 (Level1PredictionBlock): Predicts within-claim relationships.
    - prediction_block_lvl2 (Level2PredictionBlock): Predicts across-claim relationships using context.
    - non_linear_predictor (nn.Sequential): Optional predictor head for regression tasks.
    - logits_generator (LogitsGenerator): Generates logits for CPT, ICD, and TTNC tokens.

    Methods:
    --------
    initialize_target_encoders():
        Copies the weights of the context encoders to the target encoders and freezes the target encoder parameters.

    update_target_encoders():
        Updates the target encoders using exponential moving average (EMA) of the context encoder parameters.

    calculate_vicreg_loss(context_output, target_output, level):
        Calculates the VICReg loss for context and target embeddings, which includes variance, invariance, and covariance penalties.
        - Parameters:
            - context_output (Tensor): Embeddings from the context encoder. Shape: [batch_size, embedding_dim].
            - target_output (Tensor): Embeddings from the target encoder. Shape: [batch_size, embedding_dim].
            - level (str): Indicates whether it's Level 1 or Level 2 ('1' or '2').
        - Returns:
            - vicreg_loss (Tensor): VICReg loss combining variance, invariance, and covariance penalties.
            - var_loss (Tensor): Variance loss.
            - inv_loss (Tensor): Invariance loss (MSE).
    
    calculate_total_loss(vicreg_loss_lvl1, vicreg_loss_lvl2, task_loss, lvl2_weight, token_pred_loss, sae_loss=0):
        Combines VICReg, task, token prediction, and sparse autoencoder losses into a total loss.
        - Parameters:
            - vicreg_loss_lvl1 (Tensor): VICReg loss at Level 1.
            - vicreg_loss_lvl2 (Tensor): VICReg loss at Level 2.
            - task_loss (Tensor): Loss for any regression tasks (optional).
            - lvl2_weight (float): Weight for the Level 2 VICReg loss.
            - token_pred_loss (Tensor): Loss for token prediction tasks (optional).
            - sae_loss (Tensor): Sparse autoencoder reconstruction loss (optional).
        - Returns:
            - total_loss (Tensor): The combined total loss.
            - task_loss (Tensor): Task-specific loss (if applicable).

    create_multi_hot_targets(target_tokens, vocab_size, padding_idx=0):
        Converts token sequences into multi-hot vectors.
        - Parameters:
            - target_tokens (Tensor): Target token sequences. Shape: [batch_size, num_claims, num_codes_per_claim].
            - vocab_size (int): Size of the token vocabulary.
            - padding_idx (int): Index for padding tokens.
        - Returns:
            - multi_hot_targets (Tensor): Multi-hot encoded vectors. Shape: [batch_size, vocab_size].

    sample_cpt_codes(cpt_logits):
        Samples CPT codes from predicted logits using a threshold.
        - Parameters:
            - cpt_logits (Tensor): Logits for CPT codes. Shape: [batch_size, vocab_size].
        - Returns:
            - fake_cpt_codes (Tensor): Binary matrix of sampled CPT codes. Shape: [batch_size, vocab_size].

    forward(cpt_tensor, icd_tensor, ttnc_tensor, target=None, teacher_forcing=True, generation=False):
        Defines the forward pass of the model. Can be used in two modes:
        - Training (teacher forcing or not).
        - Autoregressive generation.
        - Parameters:
            - cpt_tensor (Tensor): Input CPT tokens. Shape: [batch_size, num_claims, num_codes_per_claim].
            - icd_tensor (Tensor): Input ICD tokens. Shape: [batch_size, num_claims, num_codes_per_claim].
            - ttnc_tensor (Tensor): Input TTNC tokens. Shape: [batch_size, num_claims].
            - target (Tensor, optional): Regression target for the task (if applicable).
            - teacher_forcing (bool): Whether to use teacher forcing during training.
            - generation (bool): Whether the forward pass is for autoregressive generation.
        - Returns:
            - Dict with outputs including loss, predictions, logits, etc.

    autoregressive_generation(cpt_tensor, icd_tensor, ttnc_tensor):
        Autoregressively generates CPT, ICD, and TTNC codes using predicted logits and dynamic thresholds.
        - Parameters:
            - cpt_tensor (Tensor): Input CPT tokens. Shape: [batch_size, num_claims, num_codes_per_claim].
            - icd_tensor (Tensor): Input ICD tokens. Shape: [batch_size, num_claims, num_codes_per_claim].
            - ttnc_tensor (Tensor): Input TTNC tokens. Shape: [batch_size, num_claims].
        - Returns:
            - Dict with predicted CPT, ICD, and TTNC codes.

    training_step(batch, batch_idx):
        Defines the training step for the generator.
        - Parameters:
            - batch (Tuple): Batch of input data (CPT, ICD, TTNC tokens, target).
            - batch_idx (int): Index of the current batch.
        - Returns:
            - total_loss (Tensor): The computed loss for the batch.

    on_train_batch_end(outputs, batch, batch_idx, dataloader_idx=None):
        Executes additional operations at the end of each training batch, including logging and evaluation.
        - Parameters:
            - outputs (Dict): Outputs from the forward pass.
            - batch (Tuple): Batch of input data.
            - batch_idx (int): Index of the current batch.

    configure_optimizers():
        Configures the optimizer for the generator with different learning rates and weight decay for specific parameters.
    """
    def __init__(self, config):
        config = apply_runtime_config_overrides(config)
        config.patient_representation_dim = config.embedding_dim * (
            2 if config.use_context_pooled_patient_representation else 1
        )
        super(HierarchicalClaimsModel, self).__init__()
        self.automatic_optimization = True
        self.save_hyperparameters()
        if getattr(config, 'debug_generation', False):
            print("Initializing HierarchicalClaimsModel")
            print(f"cpt_vocab_size: {config.cpt_vocab_size}")
            print(f"icd_vocab_size: {config.icd_vocab_size}")
            print(f"ttnc_vocab_size: {config.ttnc_vocab_size}")
            print(f"embedding_dim: {config.embedding_dim}")

        # Configuration parameters
        self.ema_decay = config.ema_decay
        self.use_eval_polyak_average = bool(
            getattr(config, "use_eval_polyak_average", False)
        )
        self.eval_polyak_decay = float(
            getattr(config, "eval_polyak_decay", 0.9999)
        )
        self.eval_polyak_update_interval = int(
            getattr(config, "eval_polyak_update_interval", 32)
        )
        self._eval_polyak_active = False
        self._eval_polyak_online_backup = None
        self.epsilon = config.epsilon
        self.target_var_lvl1 = config.target_var_lvl1
        self.target_var_lvl2 = config.target_var_lvl2
        self.amplification_power = config.amplification_power
        self.steps_per_epoch = config.steps_per_epoch
        self.var_penalty_scale_lvl1 = config.var_penalty_scale_lvl1
        self.cov_penalty_scale_lvl1 = config.cov_penalty_scale_lvl1
        self.var_penalty_scale_lvl2 = config.var_penalty_scale_lvl2
        self.cov_penalty_scale_lvl2 = config.cov_penalty_scale_lvl2
        self.rnn_type = config.rnn_type
        self.level_2_weight = config.level_2_weight
        self.ssl_objective_type = config.ssl_objective_type
        self.target_encoder_mode = config.target_encoder_mode
        self.target_encoder_mode_lvl1 = getattr(
            config, "target_encoder_mode_lvl1", self.target_encoder_mode
        )
        self.target_encoder_mode_lvl2 = getattr(
            config, "target_encoder_mode_lvl2", self.target_encoder_mode
        )
        self.clean_ssl_mode = getattr(config, "clean_ssl_mode", False)
        self.use_level2_dense_prediction = getattr(config, "use_level2_dense_prediction", False)
        self.observed_claim_k = getattr(config, "observed_claim_k", 0)
        self.observed_claim_loss_weight = getattr(config, "observed_claim_loss_weight", 0.25)
        self.future_claim_k = getattr(config, "future_claim_k", 0)
        self.future_claim_loss_weight = getattr(config, "future_claim_loss_weight", 0.5)
        self.future_claim_loss_decay = getattr(config, "future_claim_loss_decay", 0.5)
        self.next_claim_loss_weight = getattr(config, "next_claim_loss_weight", 1.0)
        self.use_world_model_dynamics = getattr(config, "use_world_model_dynamics", False)
        self.world_model_state_dim = getattr(
            config,
            "world_model_state_dim",
            config.patient_representation_dim,
        )
        self.world_model_future_steps = getattr(
            config,
            "world_model_future_steps",
            self.future_claim_k + 1,
        )
        self.world_model_next_weight = getattr(config, "world_model_next_weight", 1.0)
        self.world_model_summary_weight = getattr(
            config,
            "world_model_summary_weight",
            0.0,
        )
        self.use_temporal_contrastive = getattr(config, "use_temporal_contrastive", False)
        self.temporal_ssl_mode = getattr(config, "temporal_ssl_mode", "cpc")
        self.temporal_loss_weight = getattr(config, "temporal_loss_weight", 0.0)
        self.temporal_projection_dim = getattr(
            config,
            "temporal_projection_dim",
            config.output_dim,
        )
        self.temporal_temperature = getattr(config, "temporal_temperature", 0.1)
        self.temporal_context_k = getattr(config, "temporal_context_k", 4)
        self.temporal_future_steps = getattr(
            config,
            "temporal_future_steps",
            self.future_claim_k + 1,
        )
        self.use_graph_pretrained_code_embeddings = getattr(
            config,
            "use_graph_pretrained_code_embeddings",
            False,
        )
        self.graph_embedding_path = getattr(config, "graph_embedding_path", None)
        self.graph_embedding_mix = float(getattr(config, "graph_embedding_mix", 1.0))
        self.graph_transfer_cpt = bool(getattr(config, "graph_transfer_cpt", True))
        self.graph_transfer_icd = bool(getattr(config, "graph_transfer_icd", True))
        self.graph_transfer_ttnc = bool(getattr(config, "graph_transfer_ttnc", True))
        self.use_masked_next_claim_token_grounding = getattr(
            config,
            "use_masked_next_claim_token_grounding",
            False,
        )
        self.masked_next_claim_token_weight = getattr(
            config,
            "masked_next_claim_token_weight",
            0.2,
        )
        self.masked_next_claim_mask_ratio = getattr(
            config,
            "masked_next_claim_mask_ratio",
            0.5,
        )
        self.masked_next_claim_use_cpt = getattr(
            config,
            "masked_next_claim_use_cpt",
            True,
        )
        self.masked_next_claim_use_icd = getattr(
            config,
            "masked_next_claim_use_icd",
            True,
        )
        self.masked_next_claim_include_ttnc = getattr(
            config,
            "masked_next_claim_include_ttnc",
            False,
        )
        self.masked_next_claim_sort_target_tokens = getattr(
            config,
            "masked_next_claim_sort_target_tokens",
            True,
        )
        self.use_claim_prototypes = getattr(config, "use_claim_prototypes", False)
        self.claim_prototype_weight = float(
            getattr(config, "claim_prototype_weight", 0.05)
        )
        self.intermediate_sequence_supervision_weight = getattr(
            config, "intermediate_sequence_supervision_weight", 0.0
        )
        self.max_generated_tokens = config.max_generated_tokens
        self.use_predictor_head = config.use_predictor_head
        self.use_grad_print = config.use_grad_print
        self.use_level1 = config.use_level1
        self.use_composable_level1 = getattr(config, "use_composable_level1", False)
        self.level1_predictive_weight = float(
            getattr(config, "level1_predictive_weight", 1.0)
        )
        self.context_cpt_dropout_prob = float(
            getattr(config, "context_cpt_dropout_prob", 0.0)
        )
        self.context_icd_dropout_prob = float(
            getattr(config, "context_icd_dropout_prob", 0.0)
        )
        self.level1_marginal_regularizer = getattr(
            config, "level1_marginal_regularizer", "none"
        )
        self.level1_marginal_weight = getattr(config, "level1_marginal_weight", 0.1)
        self.level1_marginal_max_samples = getattr(
            config, "level1_marginal_max_samples", 128
        )
        self.sigreg_num_slices = getattr(config, "sigreg_num_slices", 256)
        self.sigreg_num_points = getattr(config, "sigreg_num_points", 17)
        self.sigreg_formulation = getattr(
            config, "sigreg_formulation", "legacy_additive"
        )
        self.sigreg_weight_lvl2 = float(
            getattr(config, "sigreg_weight_lvl2", 0.1)
        )
        self.use_levjepa_patient_views = bool(
            getattr(config, "use_levjepa_patient_views", False)
        )
        self.levjepa_num_local_views = int(
            getattr(config, "levjepa_num_local_views", 2)
        )
        self.levjepa_claim_drop_ratio = float(
            getattr(config, "levjepa_claim_drop_ratio", 0.3)
        )
        self.use_zero_target_mask = config.use_zero_target_mask
        self.use_token_prediction_head = config.use_token_prediction_head
        self.use_sparse_autoencoder = config.use_sparse_autoencoder
        self.use_gated_fusion = config.use_gated_fusion
        self.use_diffusion = getattr(config, 'use_diffusion', False)
        self.diffusion_type = getattr(config, 'diffusion_type', 'continuous')
        self.diffusion_weight = getattr(config, 'diffusion_weight', 1.0)
        self.debug_generation = getattr(config, 'debug_generation', False)
        self.mean_target_baseline_rmse = getattr(config, "mean_target_baseline_rmse", None)
        self.freeze_logvars_after_epoch = getattr(config, "freeze_logvars_after_epoch", None)
        self.cpt_vocab_size = config.cpt_vocab_size
        self.icd_vocab_size = config.icd_vocab_size
        self.ttnc_vocab_size = config.ttnc_vocab_size
        self.is_stage1_pretrain = getattr(config, 'current_stage', 'stage1') == 'stage1'
        self.cpt_id_to_token = getattr(config, "cpt_id_to_token", None)
        self.icd_id_to_token = getattr(config, "icd_id_to_token", None)
        self.ttnc_id_to_token = getattr(config, "ttnc_id_to_token", None)

        self.cpt_base_threshold = getattr(config, 'base_cpt_threshold', getattr(config, 'cpt_threshold', 0.5))
        self.icd_base_threshold = getattr(config, 'base_icd_threshold', getattr(config, 'icd_threshold', 0.5))
        self.debug_low_threshold = getattr(config, 'debug_low_threshold', False)

        # Accumulators for patient representations and targets
        # Used for one-shot cross-validation at epoch end
        self.repr_accumulator = []
        self.target_accumulator = []
        self.regression_weights = None
        self.alternate_flag = True
        # Initialize SAE loss tracking
        self.sae_loss_total = 0.0
        self.sae_loss_count = 0
        self._grad_check_done = False
        self._logvars_frozen = False
        self.ssl_lvl2_raw_total = 0.0
        self.ssl_lvl2_wgt_total = 0.0
        self.ssl_batch_count = 0

        self.threshold = nn.Parameter(torch.tensor(0.1))
        self.lambda_entropy = nn.Parameter(torch.tensor(config.lambda_entropy))
        self.ssl_objective = build_ssl_objective(config)

        # Initialize Encoders and Prediction Blocks
        if self.use_level1:
            self.context_encoder_lvl1 = Level1Encoder(
                cpt_vocab_size=config.cpt_vocab_size,
                icd_vocab_size=config.icd_vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=0,
                pooling_type=config.claim_pooling_type,
                num_heads=config.claim_pooling_num_heads,
                use_rarity=config.claim_pooling_use_rarity,
                cpt_rarity_scores=config.cpt_rarity_scores,
                icd_rarity_scores=config.icd_rarity_scores,
            )
            self.target_encoder_lvl1 = Level1Encoder(
                cpt_vocab_size=config.cpt_vocab_size,
                icd_vocab_size=config.icd_vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=0,
                pooling_type=config.claim_pooling_type,
                num_heads=config.claim_pooling_num_heads,
                use_rarity=config.claim_pooling_use_rarity,
                cpt_rarity_scores=config.cpt_rarity_scores,
                icd_rarity_scores=config.icd_rarity_scores,
            )
            self.prediction_block_lvl1 = Level1PredictionBlock(
                embedding_dim=config.embedding_dim
            )
            if self.use_composable_level1:
                self.context_level1_composer = ComposableClaimEncoder(
                    embedding_dim=config.embedding_dim,
                    ttnc_vocab_size=config.ttnc_vocab_size,
                    padding_idx=0,
                    dropout=config.dropout,
                    use_ttnc=config.ttnc_in_composer,
                )
                self.target_level1_composer = ComposableClaimEncoder(
                    embedding_dim=config.embedding_dim,
                    ttnc_vocab_size=config.ttnc_vocab_size,
                    padding_idx=0,
                    dropout=config.dropout,
                    use_ttnc=config.ttnc_in_composer,
                )
                if self.level1_marginal_regularizer == "wristband":
                    self.wristband_regularizer = WristbandGaussianRegularizer(
                        embedding_dim=config.embedding_dim,
                        sample_size=self.level1_marginal_max_samples,
                        calibration_reps=getattr(config, "wristband_calibration_reps", 16),
                    )

        self.context_encoder_lvl2 = Level2Encoder(
            cpt_vocab_size=config.cpt_vocab_size,
            icd_vocab_size=config.icd_vocab_size,
            ttnc_vocab_size=config.ttnc_vocab_size,
            embedding_dim=config.embedding_dim,
            cpt_rarity_scores=config.cpt_rarity_scores,
            icd_rarity_scores=config.icd_rarity_scores,
            ttnc_rarity_scores=config.ttnc_rarity_scores,
            use_token_rarity=config.use_token_rarity,
            use_code_attention=config.use_code_attention,
            use_variance_embeddings=config.use_variance_embeddings,
            use_aggregate_attention=config.use_aggregate_attention,
            use_component_attention=config.use_component_attention,
            dropout=config.dropout
        )
        self.target_encoder_lvl2 = Level2Encoder(
            cpt_vocab_size=config.cpt_vocab_size,
            icd_vocab_size=config.icd_vocab_size,
            ttnc_vocab_size=config.ttnc_vocab_size,
            embedding_dim=config.embedding_dim,
            cpt_rarity_scores=config.cpt_rarity_scores,
            icd_rarity_scores=config.icd_rarity_scores,
            ttnc_rarity_scores=config.ttnc_rarity_scores,
            use_token_rarity=config.use_token_rarity,
            use_code_attention=config.use_code_attention,
            use_variance_embeddings=config.use_variance_embeddings,
            use_aggregate_attention=config.use_aggregate_attention,
            use_component_attention=config.use_component_attention,
            dropout=config.dropout
        )
        
        self.prediction_block_lvl2 = Level2PredictionBlock(
            embed_dim=config.embedding_dim,
            output_dim=config.output_dim,
            ttnc_vocab_size=config.ttnc_vocab_size,
            max_seq_length=config.max_claims_len,
            padding_idx=0,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_hidden_dim=config.ff_hidden_dim,
            dropout=config.dropout,
            rnn_type=config.rnn_type,
            cpt_vocab_size=config.cpt_vocab_size,
            icd_vocab_size=config.icd_vocab_size,
            use_context_pooled_patient_representation=config.use_context_pooled_patient_representation,
            use_level2_dense_prediction=self.use_level2_dense_prediction,
            observed_claim_k=self.observed_claim_k,
            future_claim_k=self.future_claim_k,
            use_bifurcated_patient_state=getattr(config, "use_bifurcated_patient_state", False),
            use_predictive_state_bottleneck=getattr(config, "use_predictive_state_bottleneck", False),
            predictive_state_bottleneck_dim=getattr(config, "predictive_state_bottleneck_dim", None),
            use_patient_state_mean_residual=getattr(config, "use_patient_state_mean_residual", False),
            patient_state_mean_residual_weight=getattr(config, "patient_state_mean_residual_weight", 0.5),
            use_dense_decoder_bottleneck=getattr(config, "use_dense_decoder_bottleneck", False),
            dense_decoder_bottleneck_dim=getattr(config, "dense_decoder_bottleneck_dim", None),
            decoder_type=config.level2_decoder_type,
            use_ttnc=config.ttnc_in_sequence,
            use_ttnc_ordinal_embedding=config.use_ttnc_ordinal_embedding,
            ttnc_ordinal_values=_build_ttnc_ordinal_values(
                self.ttnc_id_to_token, config.ttnc_vocab_size
            ),
        )
        if self.use_levjepa_patient_views:
            # Keep projector construction from perturbing initialization of
            # unrelated optional heads in paired recipe comparisons.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 63031)
                self.levjepa_patient_projector = nn.Sequential(
                    nn.Linear(
                        config.patient_representation_dim,
                        config.levjepa_projector_hidden_dim,
                    ),
                    nn.BatchNorm1d(config.levjepa_projector_hidden_dim),
                    nn.GELU(),
                    nn.Linear(
                        config.levjepa_projector_hidden_dim,
                        config.levjepa_projector_output_dim,
                    ),
                )
        else:
            self.levjepa_patient_projector = None
        if (
            self.use_composable_level1
            and config.share_ttnc_embeddings
            and config.ttnc_in_composer
            and config.ttnc_in_sequence
        ):
            self.prediction_block_lvl2.ttnc_embedding = (
                self.context_level1_composer.ttnc_embedding
            )
        self.masked_claim_query_head = (
            MaskedClaimQueryHead(
                embed_dim=config.embedding_dim,
                output_dim=config.output_dim,
                max_seq_length=config.max_claims_len,
                num_heads=config.masked_claim_jepa_num_heads,
                dropout=config.dropout,
            )
            if config.use_masked_claim_jepa
            else None
        )
        self.multi_hypothesis_future_head = (
            MultiHypothesisFutureHead(
                input_dim=config.embedding_dim * 2,
                output_dim=config.output_dim,
                num_hypotheses=config.multi_hypothesis_count,
                dropout=config.dropout,
            )
            if config.use_multi_hypothesis_future
            else None
        )
        self.use_masked_claim_jepa = config.use_masked_claim_jepa
        self.masked_claim_jepa_weight = config.masked_claim_jepa_weight
        self.use_multi_hypothesis_future = config.use_multi_hypothesis_future
        self.multi_hypothesis_weight = config.multi_hypothesis_weight
        self.multi_hypothesis_temperature = config.multi_hypothesis_temperature
        if self.use_claim_prototypes:
            # Keep the auxiliary head's shape from perturbing initialization of
            # downstream modules.  This makes K/weight sweeps directly paired
            # with the no-prototype anchor at a fixed training seed.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(int(config.seed) + 47017)
                self.claim_prototype_objective = ClaimPrototypeObjective(
                    embedding_dim=config.output_dim,
                    num_prototypes=config.claim_prototype_count,
                    assignment_temperature=config.claim_prototype_assignment_temperature,
                    clustering_temperature=config.claim_prototype_clustering_temperature,
                    student_temperature=config.claim_prototype_student_temperature,
                    sinkhorn_iterations=config.claim_prototype_sinkhorn_iterations,
                    clustering_weight=config.claim_prototype_clustering_weight,
                )
        else:
            self.claim_prototype_objective = None
        if self.use_world_model_dynamics:
            self.world_model = ClaimsWorldModel(
                claim_dim=config.output_dim,
                state_dim=self.world_model_state_dim,
                future_steps=self.world_model_future_steps,
                dropout=config.dropout,
            )
        else:
            self.world_model = None

        if self.use_temporal_contrastive:
            self.temporal_contrastive_head = TemporalContrastiveHead(
                context_dim=config.embedding_dim,
                future_dim=config.output_dim,
                projection_dim=self.temporal_projection_dim,
                temperature=self.temporal_temperature,
                mode=self.temporal_ssl_mode,
                context_k=self.temporal_context_k,
            )
        else:
            self.temporal_contrastive_head = None

        if self.use_sparse_autoencoder:
            self.sparse_autoencoder = SparseAutoencoder(
                input_dim=config.patient_representation_dim,
                hidden_dim=config.sae_hidden_dim,
                k=config.sae_k,
            )
            if self.use_gated_fusion:
                # Map the SAE hidden representation to the patient representation
                # dimension. This ensures that the SAE embedding and the patient
                # representation have matching sizes even when
                # ``use_context_pooled_patient_representation`` doubles the
                # dimensionality.
                self.sae_to_embed = nn.Linear(
                    config.sae_hidden_dim, config.patient_representation_dim
                )
                # The gating network takes the concatenation of the patient
                # representation and SAE embedding and outputs a gating vector of
                # the same dimensionality as ``patient_representation``.
                self.gating_network = nn.Sequential(
                    nn.Linear(
                        config.patient_representation_dim * 2,
                        config.gating_hidden_dim,
                    ),
                    nn.ReLU(),
                    nn.Linear(
                        config.gating_hidden_dim, config.patient_representation_dim
                    ),
                    nn.Sigmoid(),
                )

        self.lr = config.lr
        self.adapter_lr = config.adapter_lr
        self.generator_lr = config.generator_lr
        self.optimizer_type = getattr(config, "optimizer_type", "adamw")
        self.optimizer_betas = (
            getattr(config, "optimizer_beta1", 0.9),
            getattr(config, "optimizer_beta2", 0.999),
        )
        self.weight_decay = getattr(config, "weight_decay", 1e-4)
        self.scheduler_type = getattr(config, "scheduler_type", "step")
        self.scheduler_step_size = getattr(config, "scheduler_step_size", 2)
        self.scheduler_gamma = getattr(config, "scheduler_gamma", 0.5)
        self.scheduler_t_max = getattr(config, "scheduler_t_max", None)
        self.scheduler_eta_min = getattr(config, "scheduler_eta_min", 0.0)
        self.scheduler_warmup_epochs = getattr(config, "scheduler_warmup_epochs", 0)
        self.scheduler_warmup_start_factor = getattr(
            config, "scheduler_warmup_start_factor", 0.2
        )
        self.training_epochs = getattr(config, "epochs", 1)
        self.sae_weight = getattr(config, "sae_weight", 1.0)
        self.task_loss_weight = getattr(config, "task_loss_weight", 1.0)
        self.predictor_head_source = getattr(
            config,
            "predictor_head_source",
            "context_mean_pool",
        )
        self.loss_fn = nn.MSELoss()
        self.log_vars = nn.ParameterDict({
            'ssl_lvl1': nn.Parameter(torch.tensor(0.0)),
            'ssl_lvl2': nn.Parameter(torch.tensor(0.0)),
            'world_model': nn.Parameter(torch.zeros(1)),
            'temporal_contrastive': nn.Parameter(torch.zeros(1)),
            'task': nn.Parameter(torch.zeros(1)),
            'token_pred': nn.Parameter(torch.zeros(1)),
            'masked_token_grounding': nn.Parameter(torch.zeros(1)),
            'masked_claim_jepa': nn.Parameter(torch.zeros(1)),
            'multi_hypothesis': nn.Parameter(torch.zeros(1)),
            'sae': nn.Parameter(torch.zeros(1)),
            'diffusion': nn.Parameter(torch.zeros(1)),
        })

        if self.use_predictor_head:
            predictor_input_dim = self._get_predictor_head_input_dim(config)
            self.non_linear_predictor = nn.Sequential(
                nn.Linear(predictor_input_dim, config.hidden_dim),  # First layer
                nn.ReLU(),  # Non-linearity
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),  # Second layer
                nn.ReLU(),  # Non-linearity
                nn.Linear(config.hidden_dim // 2, 1)  # Output layer for regression
            )

        if self.use_token_prediction_head:
            self.logits_generator = LogitsGenerator(config)

        if self.use_masked_next_claim_token_grounding:
            grounding_hidden_dim = (
                getattr(config, "masked_next_claim_hidden_dim", None)
                or config.hidden_dim
            )
            self.masked_claim_token_decoder = MaskedClaimTokenDecoder(
                input_dim=config.output_dim,
                hidden_dim=grounding_hidden_dim,
                cpt_vocab_size=config.cpt_vocab_size,
                icd_vocab_size=config.icd_vocab_size,
                ttnc_vocab_size=config.ttnc_vocab_size,
                max_cpt_tokens=config.max_cpt_tokens,
                max_icd_tokens=config.max_icd_tokens,
                dropout=config.dropout,
                include_ttnc=self.masked_next_claim_include_ttnc,
            )

        if self.use_graph_pretrained_code_embeddings:
            self._load_graph_pretrained_code_embeddings(self.graph_embedding_path)

        self.initialize_target_encoders()
        if self.use_composable_level1:
            # The legacy token-reencoding Level-2 encoders remain in the
            # checkpoint for backward compatibility, but they are not part of
            # the composable representation or generator-conditioning paths.
            for module in (self.context_encoder_lvl2, self.target_encoder_lvl2):
                for parameter in module.parameters():
                    parameter.requires_grad = False

        if self.use_diffusion:
            # Condition the diffusion generator on the predicted claim
            # representation from the Level 2 prediction block
            if self.diffusion_type == 'discrete':
                self.diffusion_model = DiscreteDiffusionModel(
                    config,
                    condition_dim=config.output_dim,
                )
                # Inherit the active representation embeddings. In composable
                # mode the legacy Level-2 token encoders are frozen and bypassed,
                # so copying from them would silently seed the generator with
                # untrained random weights.
                self.synchronize_generator_embeddings_from_active_encoder()
                requires_grad = getattr(config, "fine_tune_embeddings", False)
                for p in [
                    self.diffusion_model.cpt_embedding.weight,
                    self.diffusion_model.icd_embedding.weight,
                    self.diffusion_model.ttnc_embedding.weight,
                ]:
                    p.requires_grad = requires_grad
                active_embeddings = self._get_active_code_embeddings()
                for modality in ("cpt", "icd", "ttnc"):
                    assert torch.allclose(
                        getattr(self.diffusion_model, f"{modality}_embedding").weight,
                        active_embeddings[modality].weight,
                    )
            else:
                self.diffusion_model = DiffusionModel(
                    config,
                    condition_dim=config.output_dim,
                )

        if (
            getattr(config, "current_stage", "stage1") == "stage2"
            and getattr(config, "freeze_encoder_at_stage2", True)
        ):
            self.freeze_encoder(getattr(config, "encoder_unfreeze_layers", 1))
            if self.use_sparse_autoencoder:
                for p in self.sparse_autoencoder.parameters():
                    p.requires_grad = False

        if self.use_eval_polyak_average:
            self._initialize_eval_polyak_average()

    def _eval_polyak_modules(self):
        modules = []
        if self.use_level1:
            modules.append(self.context_encoder_lvl1)
        if self.use_composable_level1:
            modules.append(self.context_level1_composer)
        else:
            modules.append(self.context_encoder_lvl2)
        modules.append(self.prediction_block_lvl2)
        return modules

    def _initialize_eval_polyak_average(self):
        online_parameters = []
        seen = set()
        for module in self._eval_polyak_modules():
            for parameter in module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                online_parameters.append(parameter)
        self._eval_polyak_online_parameters = online_parameters
        self.eval_polyak_parameters = nn.ParameterList(
            [
                nn.Parameter(parameter.detach().clone(), requires_grad=False)
                for parameter in online_parameters
            ]
        )
        self.register_buffer(
            "eval_polyak_batches",
            torch.zeros((), dtype=torch.long),
        )
        self.register_buffer(
            "eval_polyak_updates",
            torch.zeros((), dtype=torch.long),
        )

    @torch.no_grad()
    def update_eval_polyak_average(self):
        if not self.use_eval_polyak_average or self._eval_polyak_active:
            return False
        self.eval_polyak_batches.add_(1)
        if int(self.eval_polyak_batches.item()) % self.eval_polyak_update_interval:
            return False
        for averaged, online in zip(
            self.eval_polyak_parameters,
            self._eval_polyak_online_parameters,
        ):
            averaged.mul_(self.eval_polyak_decay).add_(
                online.detach(), alpha=1.0 - self.eval_polyak_decay
            )
        self.eval_polyak_updates.add_(1)
        return True

    @torch.no_grad()
    def activate_eval_polyak_weights(self):
        if (
            not self.use_eval_polyak_average
            or self._eval_polyak_active
            or int(self.eval_polyak_updates.item()) == 0
        ):
            return False
        self._eval_polyak_online_backup = [
            parameter.detach().clone()
            for parameter in self._eval_polyak_online_parameters
        ]
        for online, averaged in zip(
            self._eval_polyak_online_parameters,
            self.eval_polyak_parameters,
        ):
            online.copy_(averaged)
        self._eval_polyak_active = True
        return True

    @torch.no_grad()
    def restore_online_weights(self):
        if not self._eval_polyak_active:
            return False
        for online, backup in zip(
            self._eval_polyak_online_parameters,
            self._eval_polyak_online_backup,
        ):
            online.copy_(backup)
        self._eval_polyak_online_backup = None
        self._eval_polyak_active = False
        return True

    def _unfreeze_last_n(self, module, n_layers):
        """Helper to unfreeze the last ``n_layers`` child modules of ``module``."""
        for param in module.parameters():
            param.requires_grad = False
        if n_layers <= 0:
            return
        children = list(module.children())
        if not children:
            for param in module.parameters():
                param.requires_grad = True
            return
        for child in children[-n_layers:]:
            for param in child.parameters():
                param.requires_grad = True

    def _load_graph_pretrained_code_embeddings(self, embedding_path: str):
        checkpoint = torch.load(embedding_path, map_location="cpu")
        required_keys = {"embedding_dim", "cpt_embeddings", "icd_embeddings", "ttnc_embeddings"}
        missing_keys = required_keys.difference(checkpoint.keys())
        if missing_keys:
            raise ValueError(
                f"Graph embedding checkpoint at {embedding_path!r} is missing keys: "
                f"{sorted(missing_keys)}"
            )

        embedding_dim = int(checkpoint["embedding_dim"])
        if embedding_dim != self.context_encoder_lvl2.cpt_embedding.embedding_dim:
            raise ValueError(
                "Graph embedding dim does not match model embedding_dim: "
                f"{embedding_dim} vs {self.context_encoder_lvl2.cpt_embedding.embedding_dim}."
            )

        def _copy_embedding(module, attr_name: str, tensor: torch.Tensor):
            weight = getattr(module, attr_name).weight.data
            if tuple(weight.shape) != tuple(tensor.shape):
                raise ValueError(
                    f"Shape mismatch for {attr_name}: "
                    f"expected {tuple(weight.shape)}, got {tuple(tensor.shape)}."
                )
            weight.mul_(1.0 - self.graph_embedding_mix).add_(
                tensor, alpha=self.graph_embedding_mix
            )

        cpt_embeddings = checkpoint["cpt_embeddings"].float()
        icd_embeddings = checkpoint["icd_embeddings"].float()
        ttnc_embeddings = checkpoint["ttnc_embeddings"].float()

        if self.graph_transfer_cpt:
            _copy_embedding(self.context_encoder_lvl2, "cpt_embedding", cpt_embeddings)
        if self.graph_transfer_icd:
            _copy_embedding(self.context_encoder_lvl2, "icd_embedding", icd_embeddings)
        if self.graph_transfer_ttnc:
            _copy_embedding(self.context_encoder_lvl2, "ttnc_embedding", ttnc_embeddings)
            _copy_embedding(self.prediction_block_lvl2, "ttnc_embedding", ttnc_embeddings)
            if self.use_composable_level1:
                _copy_embedding(
                    self.context_level1_composer,
                    "ttnc_embedding",
                    ttnc_embeddings,
                )

        if self.use_level1:
            if self.graph_transfer_cpt:
                _copy_embedding(self.context_encoder_lvl1, "cpt_embedding", cpt_embeddings)
            if self.graph_transfer_icd:
                _copy_embedding(self.context_encoder_lvl1, "icd_embedding", icd_embeddings)

    def freeze_encoders(self, n_layers: int = 1):
        """Freeze all encoder weights except the last ``n_layers``."""
        modules = [self.context_encoder_lvl2, self.target_encoder_lvl2]
        if self.use_level1:
            modules.extend([self.context_encoder_lvl1, self.target_encoder_lvl1])
        if self.use_composable_level1:
            modules.extend([self.context_level1_composer, self.target_level1_composer])
        for mod in modules:
            self._unfreeze_last_n(mod, n_layers)

    # Backwards compatibility alias matching documentation terminology
    def freeze_encoder(self, except_last_n_layers: int = 1):
        """Freeze encoders except the final ``except_last_n_layers`` modules."""
        self.freeze_encoders(except_last_n_layers)

    def unfreeze_encoders(self):
        """Unfreeze all encoder parameters."""
        modules = [self.context_encoder_lvl2, self.target_encoder_lvl2]
        if self.use_level1:
            modules.extend([self.context_encoder_lvl1, self.target_encoder_lvl1])
        if self.use_composable_level1:
            modules.extend([self.context_level1_composer, self.target_level1_composer])
        for mod in modules:
            for param in mod.parameters():
                param.requires_grad = True

    def initialize_target_encoders(self):
        if self.use_level1:
            for param_q, param_k in zip(
                self.context_encoder_lvl1.parameters(),
                self.target_encoder_lvl1.parameters(),
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        if self.use_composable_level1:
            for param_q, param_k in zip(
                self.context_level1_composer.parameters(),
                self.target_level1_composer.parameters(),
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        # Copy context encoder parameters to target encoders and set requires_grad=False
        for param_q, param_k in zip(
            self.context_encoder_lvl2.parameters(),
            self.target_encoder_lvl2.parameters(),
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use Kaiming Normal initialization for ReLU activations
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.GRU):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:  
                    nn.init.xavier_uniform_(param.data) 
                elif 'weight_hh' in name:  
                    nn.init.orthogonal_(param.data)  
                elif 'bias' in name:  
                    nn.init.zeros_(param.data)  
        elif isinstance(module, nn.MultiheadAttention):
            nn.init.xavier_uniform_(module.in_proj_weight)
            nn.init.zeros_(module.in_proj_bias)
            nn.init.xavier_uniform_(module.out_proj.weight)
            nn.init.zeros_(module.out_proj.bias)
        elif isinstance(module, nn.TransformerEncoderLayer):
            # Initialize the Transformer Encoder Layer
            nn.init.xavier_uniform_(module.self_attn.in_proj_weight)
            nn.init.zeros_(module.self_attn.in_proj_bias)
            nn.init.xavier_uniform_(module.self_attn.out_proj.weight)
            nn.init.zeros_(module.self_attn.out_proj.bias)

            nn.init.xavier_uniform_(module.linear1.weight)
            nn.init.zeros_(module.linear1.bias)
            nn.init.xavier_uniform_(module.linear2.weight)
            nn.init.zeros_(module.linear2.bias)

            nn.init.ones_(module.norm1.weight)
            nn.init.zeros_(module.norm1.bias)
            nn.init.ones_(module.norm2.weight)
            nn.init.zeros_(module.norm2.bias)

    def update_target_encoders(self):
        if not self._uses_ema_target("1") and not self._uses_ema_target("2"):
            return

        with torch.no_grad():
            if self.use_level1 and self._uses_ema_target("1"):
                for param_q, param_k in zip(
                    self.context_encoder_lvl1.parameters(),
                    self.target_encoder_lvl1.parameters(),
                ):
                    param_k.data = self.ema_decay * param_k.data + (1.0 - self.ema_decay) * param_q.data

            if self._uses_ema_target("2"):
                for param_q, param_k in zip(
                    self.context_encoder_lvl2.parameters(),
                    self.target_encoder_lvl2.parameters(),
                ):
                    param_k.data = self.ema_decay * param_k.data + (1.0 - self.ema_decay) * param_q.data
                if self.use_composable_level1:
                    for param_q, param_k in zip(
                        self.context_level1_composer.parameters(),
                        self.target_level1_composer.parameters(),
                    ):
                        param_k.data = (
                            self.ema_decay * param_k.data
                            + (1.0 - self.ema_decay) * param_q.data
                        )

    def _get_target_mode(self, level: str) -> str:
        if level == "1":
            return self.target_encoder_mode_lvl1
        if level == "2":
            return self.target_encoder_mode_lvl2
        raise ValueError(f"Unsupported target encoder level={level!r}")

    def _uses_ema_target(self, level: str) -> bool:
        return self._get_target_mode(level) == "ema"

    def _get_ssl_target_encoder_lvl1(self):
        return (
            self.context_encoder_lvl1
            if self._get_target_mode("1") == "shared"
            else self.target_encoder_lvl1
        )

    def _get_ssl_target_encoder_lvl2(self):
        return (
            self.context_encoder_lvl2
            if self._get_target_mode("2") == "shared"
            else self.target_encoder_lvl2
        )

    def _get_reference_encoder_lvl2(self):
        return (
            self.context_encoder_lvl2
            if self._get_target_mode("2") == "shared"
            else self.target_encoder_lvl2
        )

    def _get_active_code_embeddings(self):
        """Return embeddings from the representation path that is actually trained."""
        if self.use_composable_level1:
            return {
                "cpt": self.context_encoder_lvl1.cpt_embedding,
                "icd": self.context_encoder_lvl1.icd_embedding,
                "ttnc": self.context_level1_composer.ttnc_embedding,
            }
        reference_encoder = self._get_reference_encoder_lvl2()
        return {
            "cpt": reference_encoder.cpt_embedding,
            "icd": reference_encoder.icd_embedding,
            "ttnc": reference_encoder.ttnc_embedding,
        }

    def synchronize_generator_embeddings_from_active_encoder(self):
        """Copy trained active embeddings into an attached discrete generator.

        This is safe to call both during construction and after loading a
        Stage-1 encoder checkpoint into a Stage-2 model.
        """
        diffusion_model = getattr(self, "diffusion_model", None)
        if diffusion_model is None or not all(
            hasattr(diffusion_model, f"{modality}_embedding")
            for modality in ("cpt", "icd", "ttnc")
        ):
            return
        active_embeddings = self._get_active_code_embeddings()
        with torch.no_grad():
            for modality, source_embedding in active_embeddings.items():
                getattr(diffusion_model, f"{modality}_embedding").weight.copy_(
                    source_embedding.weight
                )

    def _get_ssl_target_composer(self):
        return (
            self.context_level1_composer
            if self._get_target_mode("2") == "shared"
            else self.target_level1_composer
        )

    def representation_sharing_contract(self):
        """Describe the effective representation path and parameter sharing.

        In the composable architecture the claim composer is the Level-2 claim
        representation function. The legacy ``Level2Encoder`` modules remain
        available for generator embedding compatibility, but do not participate
        in representation learning. Making that distinction explicit prevents
        ``target_encoder_mode_lvl2`` from being misread as a second token-level
        encoding pass.
        """
        composable = bool(self.use_composable_level1)
        return {
            "level2_claim_input": (
                "composed_level1_claim_state"
                if composable
                else "legacy_level2_token_reencoding"
            ),
            "level1_task_and_level2_input_encoder_shared": bool(
                composable and self.use_level1
            ),
            "context_target_level1_encoder_mode": (
                self._get_target_mode("1") if self.use_level1 else "inactive"
            ),
            "context_target_claim_composer_mode": (
                self._get_target_mode("2") if composable else "inactive"
            ),
            "legacy_level2_token_encoder_mode": (
                "inactive" if composable else self._get_target_mode("2")
            ),
            "level2_sequence_predictor_parameters": "online_shared",
            "level2_patient_view_objective": (
                "projected_global_local_invariance"
                if self.use_levjepa_patient_views
                else "inactive"
            ),
            "evaluation_parameter_average": (
                "polyak_shadow_no_training_forward"
                if self.use_eval_polyak_average
                else "inactive"
            ),
            "claim_prototype_target": (
                "detached_shared_composed_claim"
                if self.use_claim_prototypes
                else "inactive"
            ),
        }

    def encode_claims(
        self,
        cpt_tensor,
        icd_tensor,
        ttnc_tensor,
        *,
        target=False,
        return_components=False,
    ):
        """Encode claims through the canonical Level-2 input path."""
        if not self.use_composable_level1:
            encoder = self._get_ssl_target_encoder_lvl2() if target else self.context_encoder_lvl2
            encoded = encoder(cpt_tensor, icd_tensor, ttnc_tensor)
            if return_components:
                return encoded, None
            return encoded

        level1_encoder = self._get_ssl_target_encoder_lvl1() if target else self.context_encoder_lvl1
        composer = self._get_ssl_target_composer() if target else self.context_level1_composer
        cpt_level1, cpt_mask = level1_encoder(cpt_tensor, "cpt")
        icd_level1, icd_mask = level1_encoder(icd_tensor, "icd")
        return composer(
            cpt_level1,
            icd_level1,
            cpt_mask,
            icd_mask,
            ttnc_tensor,
            return_components=return_components,
        )

    def _compute_level1_marginal_regularizer(self, components):
        zero = next(self.parameters()).new_zeros(())
        if components is None or self.level1_marginal_regularizer == "none":
            return zero, {"cpt": zero, "icd": zero}

        losses = {}
        for modality in ("cpt", "icd"):
            marginal = components[f"{modality}_marginal"]
            mask = components[f"{modality}_mask"]
            valid = marginal.reshape(-1, marginal.size(-1))[mask.reshape(-1)]
            if self.level1_marginal_regularizer == "sigreg":
                loss = sigreg_gaussian_distance(
                    valid,
                    num_slices=self.sigreg_num_slices,
                    num_points=self.sigreg_num_points,
                    epsilon=self.epsilon,
                )
            elif self.level1_marginal_regularizer == "wristband":
                loss, _ = self.wristband_regularizer(valid)
            else:
                raise ValueError(
                    f"Unsupported Level-1 marginal regularizer: "
                    f"{self.level1_marginal_regularizer!r}"
                )
            losses[modality] = loss
        return (losses["cpt"] + losses["icd"]) / 2.0, losses

    def _extract_next_claim_prediction(self, prediction):
        if prediction.dim() == 3:
            return prediction[:, self.observed_claim_k]
        return prediction

    def _build_level2_ssl_targets(self, cpt_tensor, icd_tensor, ttnc_tensor):
        if not self.use_level2_dense_prediction:
            context_cpt = cpt_tensor[:, :-1]
            context_icd = icd_tensor[:, :-1]
            context_ttnc = ttnc_tensor[:, :-1]
            target_cpt = cpt_tensor[:, -1].unsqueeze(1)
            target_icd = icd_tensor[:, -1].unsqueeze(1)
            target_ttnc = ttnc_tensor[:, -1].unsqueeze(1)
            target_repr = self.encode_claims(
                target_cpt,
                target_icd,
                target_ttnc,
                target=True,
            ).squeeze(1)
            target_mask = target_ttnc.squeeze(1) != 0
            return {
                "context_cpt": context_cpt,
                "context_icd": context_icd,
                "context_ttnc": context_ttnc,
                "target_cpt": target_cpt,
                "target_icd": target_icd,
                "target_ttnc": target_ttnc,
                "target_repr": target_repr,
                "target_mask": target_mask,
                "slot_weights": None,
            }

        batch_size, total_seq_len, max_cpt_tokens = cpt_tensor.size()
        max_icd_tokens = icd_tensor.size(-1)
        future_slot_count = self.future_claim_k + 1
        slot_count = self.observed_claim_k + future_slot_count
        context_seq_len = max(1, total_seq_len - future_slot_count)
        context_cpt = torch.zeros(
            batch_size,
            context_seq_len,
            max_cpt_tokens,
            dtype=cpt_tensor.dtype,
            device=cpt_tensor.device,
        )
        context_icd = torch.zeros(
            batch_size,
            context_seq_len,
            max_icd_tokens,
            dtype=icd_tensor.dtype,
            device=icd_tensor.device,
        )
        context_ttnc = torch.zeros(
            batch_size,
            context_seq_len,
            dtype=ttnc_tensor.dtype,
            device=ttnc_tensor.device,
        )
        target_cpt = torch.zeros(
            batch_size,
            1,
            max_cpt_tokens,
            dtype=cpt_tensor.dtype,
            device=cpt_tensor.device,
        )
        target_icd = torch.zeros(
            batch_size,
            1,
            max_icd_tokens,
            dtype=icd_tensor.dtype,
            device=icd_tensor.device,
        )
        target_ttnc = torch.zeros(
            batch_size,
            1,
            dtype=ttnc_tensor.dtype,
            device=ttnc_tensor.device,
        )
        dense_cpt = torch.zeros(
            batch_size,
            slot_count,
            max_cpt_tokens,
            dtype=cpt_tensor.dtype,
            device=cpt_tensor.device,
        )
        dense_icd = torch.zeros(
            batch_size,
            slot_count,
            max_icd_tokens,
            dtype=icd_tensor.dtype,
            device=icd_tensor.device,
        )
        dense_ttnc = torch.zeros(
            batch_size,
            slot_count,
            dtype=ttnc_tensor.dtype,
            device=ttnc_tensor.device,
        )
        dense_mask = torch.zeros(
            batch_size,
            slot_count,
            dtype=torch.bool,
            device=ttnc_tensor.device,
        )

        sequence_valid = ttnc_tensor != 0
        for batch_index in range(batch_size):
            valid_indices = torch.nonzero(sequence_valid[batch_index], as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                continue

            available_future = min(
                future_slot_count,
                max(valid_indices.numel() - 1, 1),
            )
            context_indices = valid_indices[: valid_indices.numel() - available_future]
            future_indices = valid_indices[valid_indices.numel() - available_future :]

            if context_indices.numel() == 0 and valid_indices.numel() > 1:
                context_indices = valid_indices[:1]
                future_indices = valid_indices[1:]

            context_offset = context_seq_len - context_indices.numel()
            for context_offset_index, context_index in enumerate(context_indices.tolist()):
                dest_index = context_offset + context_offset_index
                context_cpt[batch_index, dest_index] = cpt_tensor[batch_index, context_index]
                context_icd[batch_index, dest_index] = icd_tensor[batch_index, context_index]
                context_ttnc[batch_index, dest_index] = ttnc_tensor[batch_index, context_index]

            if self.observed_claim_k > 0:
                observed_indices = context_indices[-self.observed_claim_k :]
                observed_offset = self.observed_claim_k - observed_indices.numel()
            else:
                observed_indices = context_indices.new_empty((0,))
                observed_offset = 0

            for slot_offset, context_index in enumerate(observed_indices.tolist()):
                slot_index = observed_offset + slot_offset
                dense_cpt[batch_index, slot_index] = cpt_tensor[batch_index, context_index]
                dense_icd[batch_index, slot_index] = icd_tensor[batch_index, context_index]
                dense_ttnc[batch_index, slot_index] = ttnc_tensor[batch_index, context_index]
                dense_mask[batch_index, slot_index] = True

            for future_slot_offset, future_index in enumerate(future_indices.tolist()):
                slot_index = self.observed_claim_k + future_slot_offset
                dense_cpt[batch_index, slot_index] = cpt_tensor[batch_index, future_index]
                dense_icd[batch_index, slot_index] = icd_tensor[batch_index, future_index]
                dense_ttnc[batch_index, slot_index] = ttnc_tensor[batch_index, future_index]
                dense_mask[batch_index, slot_index] = True

            if future_indices.numel() > 0:
                immediate_target_index = future_indices[0].item()
                target_cpt[batch_index, 0] = cpt_tensor[batch_index, immediate_target_index]
                target_icd[batch_index, 0] = icd_tensor[batch_index, immediate_target_index]
                target_ttnc[batch_index, 0] = ttnc_tensor[batch_index, immediate_target_index]

        slot_weights = torch.zeros(
            batch_size,
            slot_count,
            dtype=torch.float32,
            device=ttnc_tensor.device,
        )
        if self.observed_claim_k > 0:
            slot_weights[:, : self.observed_claim_k] = self.observed_claim_loss_weight
        slot_weights[:, self.observed_claim_k] = self.next_claim_loss_weight
        for future_offset in range(1, future_slot_count):
            slot_weights[:, self.observed_claim_k + future_offset] = (
                self.future_claim_loss_weight
                * (self.future_claim_loss_decay ** (future_offset - 1))
            )
        slot_weights = slot_weights * dense_mask.float()

        target_repr = self.encode_claims(
            dense_cpt,
            dense_icd,
            dense_ttnc,
            target=True,
        )
        return {
            "context_cpt": context_cpt,
            "context_icd": context_icd,
            "context_ttnc": context_ttnc,
            "target_cpt": target_cpt,
            "target_icd": target_icd,
            "target_ttnc": target_ttnc,
            "target_repr": target_repr,
            "target_mask": dense_mask,
            "slot_weights": slot_weights,
        }

    def _apply_context_modality_dropout(
        self,
        cpt_tensor,
        icd_tensor,
        ttnc_tensor,
    ):
        """Drop one observed code family from Level-2 context claims only.

        Corruption is restricted to claims where CPT and ICD are both present,
        which guarantees that a valid claim never loses its final code family.
        The CPT draw takes precedence and ICD is sampled from the remaining
        claims, so both modalities cannot be dropped together.
        """
        if not self.training:
            return cpt_tensor, icd_tensor

        cpt_probability = self.context_cpt_dropout_prob
        icd_probability = self.context_icd_dropout_prob
        if cpt_probability <= 0.0 and icd_probability <= 0.0:
            return cpt_tensor, icd_tensor

        claim_present = ttnc_tensor.ne(0)
        cpt_present = cpt_tensor.ne(0).any(dim=-1)
        icd_present = icd_tensor.ne(0).any(dim=-1)
        both_present = claim_present & cpt_present & icd_present

        cpt_drop = torch.zeros_like(both_present)
        if cpt_probability > 0.0:
            cpt_drop = (
                torch.rand(both_present.shape, device=both_present.device)
                < cpt_probability
            ) & both_present

        icd_drop = torch.zeros_like(both_present)
        if icd_probability > 0.0:
            icd_drop = (
                torch.rand(both_present.shape, device=both_present.device)
                < icd_probability
            ) & both_present & ~cpt_drop

        if cpt_drop.any():
            cpt_tensor = cpt_tensor.clone()
            cpt_tensor[cpt_drop] = 0
        if icd_drop.any():
            icd_tensor = icd_tensor.clone()
            icd_tensor[icd_drop] = 0
        return cpt_tensor, icd_tensor

    def _sample_levjepa_local_history(
        self,
        cpt_tensor,
        icd_tensor,
        ttnc_tensor,
    ):
        """Thin a history view while preserving its latest observed claim.

        The corruption is an encoder observation, not a reconstruction mask:
        dropped claims are removed from every input stream and receive no
        token-level target. Each retained claim remains otherwise intact so a
        naturally partial CPT- or ICD-only claim is never fabricated.
        """
        valid_claims = ttnc_tensor.ne(0)
        if self.levjepa_claim_drop_ratio <= 0.0:
            keep_claims = valid_claims.clone()
        else:
            keep_claims = (
                torch.rand(valid_claims.shape, device=ttnc_tensor.device)
                >= self.levjepa_claim_drop_ratio
            ) & valid_claims

        has_valid_claim = valid_claims.any(dim=1)
        if has_valid_claim.any():
            last_valid = (
                valid_claims.size(1)
                - 1
                - valid_claims.flip(1).to(torch.int64).argmax(dim=1)
            )
            batch_indices = torch.arange(
                valid_claims.size(0), device=ttnc_tensor.device
            )[has_valid_claim]
            keep_claims[batch_indices, last_valid[has_valid_claim]] = True

        dropped_claims = valid_claims & ~keep_claims
        local_cpt = cpt_tensor.clone()
        local_icd = icd_tensor.clone()
        local_ttnc = ttnc_tensor.clone()
        local_cpt[dropped_claims] = 0
        local_icd[dropped_claims] = 0
        local_ttnc[dropped_claims] = 0
        return local_cpt, local_icd, local_ttnc, keep_claims

    def _compute_levjepa_patient_view_loss(
        self,
        global_patient_representation,
        context_cpt,
        context_icd,
        context_ttnc,
    ):
        """Compute projected global/local invariance plus additive SIGReg."""
        patient_views = [global_patient_representation]
        retained_fractions = []
        valid_claim_count = context_ttnc.ne(0).sum().clamp(min=1).to(
            global_patient_representation.dtype
        )

        for _ in range(self.levjepa_num_local_views):
            local_cpt, local_icd, local_ttnc, keep_claims = (
                self._sample_levjepa_local_history(
                    context_cpt,
                    context_icd,
                    context_ttnc,
                )
            )
            local_claims = self.encode_claims(
                local_cpt,
                local_icd,
                local_ttnc,
            )
            local_patient, _ = self.prediction_block_lvl2(
                local_claims,
                local_ttnc,
            )
            patient_views.append(local_patient)
            retained_fractions.append(
                keep_claims.sum().to(global_patient_representation.dtype)
                / valid_claim_count
            )

        stacked_views = torch.stack(patient_views, dim=0)
        projected_views = self.levjepa_patient_projector(
            stacked_views.reshape(-1, stacked_views.size(-1))
        ).reshape(
            stacked_views.size(0),
            stacked_views.size(1),
            -1,
        )
        global_projection = projected_views[0]
        local_projections = projected_views[1:]
        invariance = (
            local_projections - global_projection.unsqueeze(0)
        ).square().mean()

        per_view_sigreg = torch.stack(
            [
                sigreg_gaussian_distance(
                    view,
                    num_slices=self.sigreg_num_slices,
                    num_points=self.sigreg_num_points,
                    epsilon=self.epsilon,
                    formulation="levjepa_additive",
                )
                for view in projected_views
            ]
        )
        sigreg_raw = per_view_sigreg.mean()
        regularizer = self.sigreg_weight_lvl2 * sigreg_raw
        retained_fraction = torch.stack(retained_fractions).mean()
        return {
            "total": invariance + regularizer,
            "invariance": invariance,
            "regularizer": regularizer,
            "sigreg_raw": sigreg_raw,
            "retained_claim_fraction": retained_fraction,
            "global_projection": global_projection,
            "local_projections": local_projections,
        }

    def _compute_embedding_variance(self, prediction, mask=None):
        if mask is None:
            flattened = prediction.reshape(-1, prediction.size(-1))
        else:
            flattened = prediction.reshape(-1, prediction.size(-1))[mask.reshape(-1).bool()]

        if flattened.numel() == 0 or flattened.size(0) < 2:
            return prediction.new_zeros(())

        return torch.var(flattened, dim=0).mean()

    def _compute_claim_slot_mse(self, prediction, target, mask):
        zero = prediction.new_zeros(())
        if prediction.dim() != 3:
            if mask.dim() > 1:
                mask = mask.squeeze(-1)
            if target.dim() > 2:
                target = target.squeeze(1)
            next_loss = F.mse_loss(
                prediction[mask],
                target[mask],
            ) if mask.any() else zero
            return zero, next_loss, zero

        observed_mask = mask[:, : self.observed_claim_k]
        next_mask = mask[:, self.observed_claim_k]
        future_mask = mask[:, self.observed_claim_k + 1 :]

        if observed_mask.any():
            observed_loss = F.mse_loss(
                prediction[:, : self.observed_claim_k][observed_mask],
                target[:, : self.observed_claim_k][observed_mask],
            )
        else:
            observed_loss = zero

        if next_mask.any():
            next_loss = F.mse_loss(
                prediction[:, self.observed_claim_k][next_mask],
                target[:, self.observed_claim_k][next_mask],
            )
        else:
            next_loss = zero

        if future_mask.numel() > 0 and future_mask.any():
            future_loss = F.mse_loss(
                prediction[:, self.observed_claim_k + 1 :][future_mask],
                target[:, self.observed_claim_k + 1 :][future_mask],
            )
        else:
            future_loss = zero

        return observed_loss, next_loss, future_loss

    def _build_world_model_targets(self, target_repr, target_mask):
        if target_repr.dim() != 3:
            future_targets = target_repr.unsqueeze(1)
            future_mask = target_mask.unsqueeze(-1) if target_mask.dim() == 1 else target_mask
        else:
            start = self.observed_claim_k
            end = start + self.world_model_future_steps
            future_targets = target_repr[:, start:end]
            future_mask = target_mask[:, start:end]

        next_target = future_targets[:, 0]
        next_mask = future_mask[:, 0]

        summary_mask = future_mask.any(dim=1)
        valid_counts = future_mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        summary_target = (future_targets * future_mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts

        return {
            "future_targets": future_targets,
            "future_mask": future_mask,
            "next_target": next_target,
            "next_mask": next_mask,
            "summary_target": summary_target,
            "summary_mask": summary_mask,
        }

    def _compute_world_model_loss(self, world_model_outputs, world_model_targets):
        zero = world_model_outputs["world_model_next_claim"].new_zeros(())

        next_mask = world_model_targets["next_mask"]
        if next_mask.any():
            next_loss = F.mse_loss(
                world_model_outputs["world_model_next_claim"][next_mask],
                world_model_targets["next_target"][next_mask],
            )
        else:
            next_loss = zero

        summary_mask = world_model_targets["summary_mask"]
        if summary_mask.any():
            summary_loss = F.mse_loss(
                world_model_outputs["world_model_future_summary"][summary_mask],
                world_model_targets["summary_target"][summary_mask],
            )
        else:
            summary_loss = zero

        total_loss = (
            next_loss * self.world_model_next_weight
            + summary_loss * self.world_model_summary_weight
        )

        return total_loss, next_loss, summary_loss

    def _build_temporal_contrastive_targets(self, target_repr, target_mask):
        if target_repr.dim() != 3:
            future_targets = target_repr.unsqueeze(1)
            future_mask = (
                target_mask.unsqueeze(1)
                if target_mask.dim() == 1
                else target_mask
            )
            return future_targets, future_mask

        start = self.observed_claim_k
        end = start + self.temporal_future_steps
        future_targets = target_repr[:, start:end]
        future_mask = target_mask[:, start:end]
        return future_targets, future_mask

    def _compute_temporal_contrastive_loss(
        self,
        sequence_output,
        context_ttnc,
        target_repr,
        target_mask,
    ):
        future_targets, future_mask = self._build_temporal_contrastive_targets(
            target_repr,
            target_mask,
        )
        context_mask = context_ttnc != 0
        return self.temporal_contrastive_head(
            sequence_output,
            context_mask,
            future_targets,
            future_mask,
        )

    def _compute_masked_next_claim_token_grounding_loss(
        self,
        next_claim_latent,
        target_cpt,
        target_icd,
        target_ttnc,
    ):
        zero = next_claim_latent.new_zeros(())
        sorted_target_cpt = target_cpt
        sorted_target_icd = target_icd
        if self.masked_next_claim_sort_target_tokens:
            sorted_target_cpt = self._sort_token_targets(
                target_cpt,
                id_to_token=self.cpt_id_to_token,
            )
            sorted_target_icd = self._sort_token_targets(
                target_icd,
                id_to_token=self.icd_id_to_token,
            )

        cpt_logits, icd_logits, ttnc_logits = self.masked_claim_token_decoder(
            next_claim_latent
        )
        cpt_loss = zero
        icd_loss = zero
        cpt_mask = None
        cpt_valid_mask = None
        icd_mask = None
        icd_valid_mask = None
        if self.masked_next_claim_use_cpt:
            cpt_mask, cpt_valid_mask = self._sample_mask_positions(
                sorted_target_cpt,
                self.masked_next_claim_mask_ratio,
            )
            cpt_loss = self._masked_token_cross_entropy(
                cpt_logits,
                sorted_target_cpt,
                cpt_mask,
            )
        if self.masked_next_claim_use_icd:
            icd_mask, icd_valid_mask = self._sample_mask_positions(
                sorted_target_icd,
                self.masked_next_claim_mask_ratio,
            )
            icd_loss = self._masked_token_cross_entropy(
                icd_logits,
                sorted_target_icd,
                icd_mask,
            )

        ttnc_loss = zero
        ttnc_mask = None
        ttnc_valid_mask = None
        if self.masked_next_claim_include_ttnc and ttnc_logits is not None:
            ttnc_targets = target_ttnc.unsqueeze(1)
            ttnc_mask, ttnc_valid_mask = self._sample_mask_positions(
                ttnc_targets,
                self.masked_next_claim_mask_ratio,
            )
            ttnc_loss = self._masked_token_cross_entropy(
                ttnc_logits.unsqueeze(1),
                ttnc_targets,
                ttnc_mask,
            )

        def _mask_fraction(mask_tensor, valid_tensor):
            if mask_tensor is None or valid_tensor is None or not valid_tensor.any():
                return zero
            return mask_tensor.float().sum() / valid_tensor.float().sum().clamp(min=1.0)

        total_loss = cpt_loss + icd_loss + ttnc_loss
        return {
            "total": total_loss,
            "cpt_loss": cpt_loss,
            "icd_loss": icd_loss,
            "ttnc_loss": ttnc_loss,
            "cpt_mask_fraction": _mask_fraction(cpt_mask, cpt_valid_mask),
            "icd_mask_fraction": _mask_fraction(icd_mask, icd_valid_mask),
            "ttnc_mask_fraction": _mask_fraction(ttnc_mask, ttnc_valid_mask),
            "cpt_logits": cpt_logits,
            "icd_logits": icd_logits,
            "ttnc_logits": ttnc_logits,
            "target_cpt": sorted_target_cpt,
            "target_icd": sorted_target_icd,
        }

    def _compute_ssl_loss(
        self,
        prediction,
        target,
        level,
        mask=None,
        sample_weights=None,
    ):
        ssl_metrics = self.ssl_objective.compute(
            prediction,
            target,
            mask=mask,
            level=level,
            sample_weights=sample_weights,
        )
        diagnostics = ssl_metrics.get("diagnostics", {})
        zero = prediction.new_zeros(())
        return {
            "total": ssl_metrics["total"],
            "predictive": ssl_metrics["predictive"],
            "regularizer": ssl_metrics["regularizer"],
            "variance": diagnostics.get("variance", zero),
            "covariance": diagnostics.get("covariance", zero),
            "sigreg_raw": diagnostics.get("sigreg_raw", zero),
        }

    def calculate_total_loss(
        self,
        ssl_loss_lvl1,
        ssl_loss_lvl2,
        task_loss,
        lvl2_weight,
        token_pred_loss,
        world_model_loss=0,
        temporal_contrastive_loss=0,
        masked_token_grounding_loss=0,
        masked_claim_jepa_loss=0,
        multi_hypothesis_loss=0,
        sae_loss=0,
        diffusion_loss=0,
    ):
        terms = []
        if self.use_level1:
            terms.append(("ssl_lvl1", ssl_loss_lvl1))
        if lvl2_weight > 0:
            terms.append(("ssl_lvl2", ssl_loss_lvl2 * lvl2_weight))
        if self.use_world_model_dynamics:
            terms.append(("world_model", world_model_loss))
        if self.use_temporal_contrastive and self.temporal_loss_weight > 0:
            terms.append(
                (
                    "temporal_contrastive",
                    temporal_contrastive_loss * self.temporal_loss_weight,
                )
            )
        if self.use_predictor_head and self.task_loss_weight > 0:
            terms.append(("task", task_loss * self.task_loss_weight))
        if self.use_token_prediction_head:
            terms.append(("token_pred", token_pred_loss))
        if self.use_masked_next_claim_token_grounding and self.masked_next_claim_token_weight > 0:
            terms.append(
                (
                    "masked_token_grounding",
                    masked_token_grounding_loss * self.masked_next_claim_token_weight,
                )
            )
        if self.use_masked_claim_jepa and self.masked_claim_jepa_weight > 0:
            terms.append(
                ("masked_claim_jepa", masked_claim_jepa_loss * self.masked_claim_jepa_weight)
            )
        if self.use_multi_hypothesis_future and self.multi_hypothesis_weight > 0:
            terms.append(
                ("multi_hypothesis", multi_hypothesis_loss * self.multi_hypothesis_weight)
            )
        if self.use_sparse_autoencoder and self.sae_weight > 0:
            terms.append(("sae", sae_loss * self.sae_weight))
        if self.use_diffusion and self.diffusion_weight > 0:
            terms.append(("diffusion", diffusion_loss * self.diffusion_weight))

        if not terms:
            return torch.tensor(0.0, device=self.device), task_loss

        clamped_log_vars = {
            key: torch.clamp(value, min=-5, max=5)
            for key, value in self.log_vars.items()
        }
        total_loss = torch.tensor(0.0, device=self.device)

        for name, raw_loss in terms:
            precision = torch.exp(-clamped_log_vars[name])
            total_loss = (
                total_loss
                + raw_loss * precision
                + clamped_log_vars[name]
            )

        # Average standard homoscedastic uncertainty terms to preserve the
        # historical loss scale without normalizing by total precision. The
        # former precision normalization made a shared shift of every log-var
        # invisible to the data losses and changed all objective weights when a
        # new auxiliary task was enabled.
        total_loss = total_loss / len(terms)
        return total_loss, task_loss

    def create_multi_hot_targets(self, target_tokens, vocab_size, padding_idx=0):
        # target_tokens: [batch_size, num_claims, num_codes_per_claim]
        batch_size, num_claims, num_codes_per_claim = target_tokens.shape
        device = target_tokens.device

        # Flatten the targets
        target_tokens = target_tokens.view(batch_size, -1)  # [batch_size, num_claims * num_codes_per_claim]

        # Create multi-hot target tensor using scatter
        multi_hot_targets = torch.zeros(batch_size, vocab_size, device=device)
        multi_hot_targets.scatter_(1, target_tokens, 1.0)

        # Ensure padding indices are not set
        multi_hot_targets[:, padding_idx] = 0.0

        return multi_hot_targets

    def _sort_token_targets(self, target_tokens, id_to_token=None, padding_idx=0):
        sorted_tokens = torch.zeros_like(target_tokens)
        for batch_index in range(target_tokens.size(0)):
            valid_tokens = target_tokens[batch_index][
                target_tokens[batch_index] != padding_idx
            ].tolist()
            if not valid_tokens:
                continue

            if id_to_token is None:
                valid_tokens = sorted(valid_tokens)
            else:
                valid_tokens = sorted(
                    valid_tokens,
                    key=lambda token_id: id_to_token.get(int(token_id), str(int(token_id))),
                )

            sorted_tokens[batch_index, : len(valid_tokens)] = torch.tensor(
                valid_tokens,
                dtype=target_tokens.dtype,
                device=target_tokens.device,
            )
        return sorted_tokens

    def _sample_mask_positions(self, target_tokens, mask_ratio, padding_idx=0):
        valid_mask = target_tokens != padding_idx
        sampled_mask = (
            torch.rand(
                target_tokens.shape,
                device=target_tokens.device,
            ) < mask_ratio
        ) & valid_mask

        needs_fallback = valid_mask.any(dim=1) & ~sampled_mask.any(dim=1)
        fallback_indices = torch.nonzero(needs_fallback, as_tuple=False).squeeze(1)
        for batch_index in fallback_indices.tolist():
            valid_positions = torch.nonzero(
                valid_mask[batch_index],
                as_tuple=False,
            ).squeeze(1)
            random_position = valid_positions[
                torch.randint(
                    valid_positions.numel(),
                    (1,),
                    device=target_tokens.device,
                )
            ]
            sampled_mask[batch_index, random_position] = True

        return sampled_mask, valid_mask

    def _masked_token_cross_entropy(self, logits, target_tokens, mask):
        if not mask.any():
            return logits.new_zeros(())

        masked_logits = logits[mask]
        masked_targets = target_tokens[mask]
        return F.cross_entropy(masked_logits, masked_targets)


    def multi_hot_to_indices_tensor(self, multi_hot_tensor):
        indices_list = []
        max_length = 0
        for i in range(multi_hot_tensor.size(0)):
            indices = torch.nonzero(multi_hot_tensor[i]).squeeze(1)
            if len(indices) == 0:
                indices = torch.tensor([0], device=multi_hot_tensor.device)
            indices_list.append(indices)
            max_length = max(max_length, len(indices))
        # Pad sequences to max_length
        indices_padded = torch.full((multi_hot_tensor.size(0), max_length), 0, dtype=torch.long, device=multi_hot_tensor.device)
        for i, indices in enumerate(indices_list):
            indices_padded[i, :len(indices)] = indices
        # Add claim dimension
        indices_padded = indices_padded.unsqueeze(1)  # [batch_size, 1, max_length]
        return indices_padded

    def _get_predictor_head_input_dim(self, config):
        if self.predictor_head_source in {"context_mean_pool", "context_max_pool"}:
            return config.embedding_dim
        if self.predictor_head_source in {
            "context_pooled",
            "patient_representation_pre_sae",
            "patient_representation",
        }:
            return config.patient_representation_dim
        raise ValueError(
            f"Unsupported predictor_head_source={self.predictor_head_source!r}"
        )

    def _select_predictor_head_input(
        self,
        sequence_aux,
        patient_representation_pre_sae,
        patient_representation,
    ):
        source_map = {
            "context_mean_pool": sequence_aux["context_mean_pool"],
            "context_max_pool": sequence_aux["context_max_pool"],
            "context_pooled": sequence_aux["context_pooled"],
            "patient_representation_pre_sae": patient_representation_pre_sae,
            "patient_representation": patient_representation,
        }
        try:
            return source_map[self.predictor_head_source]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported predictor_head_source={self.predictor_head_source!r}"
            ) from exc

    def forward(self, cpt_tensor, icd_tensor, ttnc_tensor, target=None, teacher_forcing=True, generation=False):
        if generation:
            return self.autoregressive_generation(cpt_tensor, icd_tensor, ttnc_tensor)
        else:
            return self.training_forward(cpt_tensor, icd_tensor, ttnc_tensor, target, teacher_forcing)

    def training_forward(self, cpt_tensor, icd_tensor, ttnc_tensor, target, teacher_forcing=True):
        zero = torch.tensor(0.0, device=self.device)
        # Handle Level 1 Encoding and Prediction
        if self.use_level1:
            target_encoder_lvl1 = self._get_ssl_target_encoder_lvl1()
            # Level 1: Within-claims prediction
            # --- First Pair: CPT as context, ICD as target ---
            context_lvl1_cpt, mask_cpt = self.context_encoder_lvl1(cpt_tensor, 'cpt')
            target_lvl1_icd, mask_icd_target = target_encoder_lvl1(icd_tensor, 'icd')

            prediction_lvl1_cpt = self.prediction_block_lvl1(context_lvl1_cpt, mask_cpt)

            batch_size, num_claims, total_embedding_dim = prediction_lvl1_cpt.shape

            # Reshape
            prediction_lvl1_cpt_flat = prediction_lvl1_cpt.view(batch_size * num_claims, total_embedding_dim)
            target_lvl1_icd_flat = target_lvl1_icd.view(batch_size * num_claims, total_embedding_dim)

            # Apply combined mask
            mask_cpt_flat = mask_cpt.view(batch_size * num_claims)
            mask_icd_target_flat = mask_icd_target.view(batch_size * num_claims)
            combined_mask_cpt = mask_cpt_flat & mask_icd_target_flat

            prediction_lvl1_cpt_flat = prediction_lvl1_cpt_flat[combined_mask_cpt]
            target_lvl1_icd_flat = target_lvl1_icd_flat[combined_mask_cpt]

            # Compute loss
            ssl_metrics_lvl1_cpt = self._compute_ssl_loss(
                prediction_lvl1_cpt_flat,
                target_lvl1_icd_flat,
                level="1",
            )

            # Second Pair: ICD as context, CPT as target
            # Pass icd_tensor to context_encoder_lvl1 and cpt_tensor to target_encoder_lvl1
            context_lvl1_icd, mask_icd = self.context_encoder_lvl1(icd_tensor, 'icd')
            target_lvl1_cpt, mask_cpt_target = target_encoder_lvl1(cpt_tensor, 'cpt')

            prediction_lvl1_icd = self.prediction_block_lvl1(context_lvl1_icd, mask_icd)

            batch_size, num_claims, total_embedding_dim = prediction_lvl1_icd.shape

            # Reshape
            prediction_lvl1_icd_flat = prediction_lvl1_icd.view(batch_size * num_claims, total_embedding_dim)
            target_lvl1_cpt_flat = target_lvl1_cpt.view(batch_size * num_claims, total_embedding_dim)

            # Apply combined mask
            mask_icd_flat = mask_icd.view(batch_size * num_claims)
            mask_cpt_target_flat = mask_cpt_target.view(batch_size * num_claims)
            combined_mask_icd = mask_icd_flat & mask_cpt_target_flat

            prediction_lvl1_icd_flat = prediction_lvl1_icd_flat[combined_mask_icd]
            target_lvl1_cpt_flat = target_lvl1_cpt_flat[combined_mask_icd]

            # Compute loss
            ssl_metrics_lvl1_icd = self._compute_ssl_loss(
                prediction_lvl1_icd_flat,
                target_lvl1_cpt_flat,
                level="1",
            )

            ssl_loss_lvl1 = (
                ssl_metrics_lvl1_cpt["total"] + ssl_metrics_lvl1_icd["total"]
            ) / 2
            level1_predictive_objective_raw = ssl_loss_lvl1
            ssl_predictive_lvl1 = (
                ssl_metrics_lvl1_cpt["predictive"] + ssl_metrics_lvl1_icd["predictive"]
            ) / 2
            ssl_regularizer_lvl1 = (
                ssl_metrics_lvl1_cpt["regularizer"] + ssl_metrics_lvl1_icd["regularizer"]
            ) / 2
            var_loss_lvl1 = (
                ssl_metrics_lvl1_cpt["variance"] + ssl_metrics_lvl1_icd["variance"]
            ) / 2
            cov_loss_lvl1 = (
                ssl_metrics_lvl1_cpt["covariance"] + ssl_metrics_lvl1_icd["covariance"]
            ) / 2
            sigreg_raw_lvl1 = (
                ssl_metrics_lvl1_cpt["sigreg_raw"] + ssl_metrics_lvl1_icd["sigreg_raw"]
            ) / 2
            embedding_variance_lvl1 = (
                self._compute_embedding_variance(prediction_lvl1_cpt_flat) +
                self._compute_embedding_variance(prediction_lvl1_icd_flat)
            ) / 2
            ssl_loss_lvl1 = ssl_loss_lvl1 * self.level1_predictive_weight
            ssl_predictive_lvl1 = (
                ssl_predictive_lvl1 * self.level1_predictive_weight
            )
            ssl_regularizer_lvl1 = (
                ssl_regularizer_lvl1 * self.level1_predictive_weight
            )
            var_loss_lvl1 = var_loss_lvl1 * self.level1_predictive_weight
            cov_loss_lvl1 = cov_loss_lvl1 * self.level1_predictive_weight
            sigreg_raw_lvl1 = sigreg_raw_lvl1 * self.level1_predictive_weight
        else:
            # Handle case when Level 1 is not used
            ssl_loss_lvl1 = zero
            ssl_predictive_lvl1 = zero
            ssl_regularizer_lvl1 = zero
            var_loss_lvl1 = zero
            cov_loss_lvl1 = zero
            sigreg_raw_lvl1 = zero
            embedding_variance_lvl1 = zero
            level1_predictive_objective_raw = zero

        level2_targets = self._build_level2_ssl_targets(cpt_tensor, icd_tensor, ttnc_tensor)
        context_cpt = level2_targets["context_cpt"]
        context_icd = level2_targets["context_icd"]
        context_ttnc = level2_targets["context_ttnc"]
        context_cpt, context_icd = self._apply_context_modality_dropout(
            context_cpt,
            context_icd,
            context_ttnc,
        )
        target_cpt = level2_targets["target_cpt"]
        target_icd = level2_targets["target_icd"]
        target_ttnc = level2_targets["target_ttnc"]
        target_lvl2 = level2_targets["target_repr"]
        target_mask_lvl2 = level2_targets["target_mask"]
        lvl2_slot_weights = level2_targets["slot_weights"]

        context_lvl2, level1_components = self.encode_claims(
            context_cpt,
            context_icd,
            context_ttnc,
            return_components=True,
        )
        level1_marginal_raw, level1_marginal_by_modality = (
            self._compute_level1_marginal_regularizer(level1_components)
        )
        level1_marginal_loss = level1_marginal_raw * self.level1_marginal_weight
        if self.use_composable_level1:
            ssl_loss_lvl1 = ssl_loss_lvl1 + level1_marginal_loss
            ssl_regularizer_lvl1 = ssl_regularizer_lvl1 + level1_marginal_loss
        sequence_context_lvl2 = context_lvl2
        masked_claim_positions = torch.zeros(
            context_lvl2.size(0), dtype=torch.long, device=context_lvl2.device
        )
        masked_claim_target = torch.zeros(
            context_lvl2.size(0), context_lvl2.size(-1), device=context_lvl2.device
        )
        masked_claim_active = torch.zeros(
            context_lvl2.size(0), dtype=torch.bool, device=context_lvl2.device
        )
        if self.use_masked_claim_jepa:
            valid_claims = context_ttnc.ne(0)
            masked_claim_active = valid_claims.any(dim=1)
            last_valid = (
                valid_claims.size(1)
                - 1
                - valid_claims.flip(1).to(torch.int64).argmax(dim=1)
            )
            candidates = valid_claims.clone()
            candidates[
                torch.arange(candidates.size(0), device=candidates.device), last_valid
            ] = False
            no_candidate = ~candidates.any(dim=1)
            candidates[no_candidate] = valid_claims[no_candidate]
            no_valid_claim = ~candidates.any(dim=1)
            candidates[no_valid_claim, 0] = True
            masked_claim_positions = torch.multinomial(
                candidates.to(torch.float32), 1
            ).squeeze(1)
            batch_indices = torch.arange(context_lvl2.size(0), device=context_lvl2.device)
            masked_claim_target = context_lvl2[
                batch_indices, masked_claim_positions
            ].detach()
            sequence_context_lvl2 = context_lvl2.clone()
            sequence_context_lvl2[batch_indices, masked_claim_positions] = 0
        patient_representation, prediction_lvl2, sequence_aux = self.prediction_block_lvl2(
            sequence_context_lvl2,
            context_ttnc,
            return_aux=True,
        )
        masked_claim_prediction = torch.zeros_like(masked_claim_target)
        masked_claim_jepa_loss = zero
        if self.use_masked_claim_jepa:
            masked_claim_prediction = self.masked_claim_query_head(
                sequence_aux["sequence_output"],
                context_ttnc.eq(0),
                masked_claim_positions,
            )
            if masked_claim_active.any():
                masked_claim_jepa_loss = F.mse_loss(
                    masked_claim_prediction[masked_claim_active],
                    masked_claim_target[masked_claim_active],
                )
        patient_representation_pre_sae = patient_representation
        next_claim_latent = self._extract_next_claim_prediction(prediction_lvl2)
        logit_context = next_claim_latent

        embedding_variance_lvl2 = self._compute_embedding_variance(
            prediction_lvl2,
            mask=target_mask_lvl2,
        )
        ssl_metrics_lvl2 = self._compute_ssl_loss(
            prediction_lvl2,
            target_lvl2,
            level="2",
            mask=target_mask_lvl2,
            sample_weights=lvl2_slot_weights,
        )
        ssl_loss_lvl2 = ssl_metrics_lvl2["total"]
        ssl_predictive_lvl2 = ssl_metrics_lvl2["predictive"]
        ssl_regularizer_lvl2 = ssl_metrics_lvl2["regularizer"]
        var_loss_lvl2 = ssl_metrics_lvl2["variance"]
        cov_loss_lvl2 = ssl_metrics_lvl2["covariance"]
        sigreg_raw_lvl2 = ssl_metrics_lvl2["sigreg_raw"]
        levjepa_patient_view_loss = zero
        levjepa_invariance_loss = zero
        levjepa_sigreg_raw = zero
        levjepa_retained_claim_fraction = zero
        levjepa_global_projection = None
        levjepa_local_projections = None
        if self.use_levjepa_patient_views:
            levjepa_metrics = self._compute_levjepa_patient_view_loss(
                patient_representation_pre_sae,
                context_cpt,
                context_icd,
                context_ttnc,
            )
            levjepa_patient_view_loss = levjepa_metrics["total"]
            levjepa_invariance_loss = levjepa_metrics["invariance"]
            levjepa_sigreg_raw = levjepa_metrics["sigreg_raw"]
            levjepa_retained_claim_fraction = levjepa_metrics[
                "retained_claim_fraction"
            ]
            levjepa_global_projection = levjepa_metrics["global_projection"]
            levjepa_local_projections = levjepa_metrics["local_projections"]

            # In this opt-in recipe the Level-2 objective is view agreement,
            # not next-claim prediction. Existing prediction outputs remain
            # available for downstream probes and Stage-2 generators.
            ssl_loss_lvl2 = levjepa_patient_view_loss
            ssl_predictive_lvl2 = levjepa_invariance_loss
            ssl_regularizer_lvl2 = levjepa_metrics["regularizer"]
            sigreg_raw_lvl2 = levjepa_sigreg_raw
            embedding_variance_lvl2 = self._compute_embedding_variance(
                levjepa_global_projection
            )

        multi_hypothesis_predictions = torch.empty(
            context_lvl2.size(0), 0, context_lvl2.size(-1), device=context_lvl2.device
        )
        multi_hypothesis_loss = zero
        multi_hypothesis_oracle_mse = zero
        if self.use_multi_hypothesis_future:
            multi_hypothesis_predictions = self.multi_hypothesis_future_head(
                sequence_aux["context_pooled"]
            )
            if target_lvl2.ndim == 3:
                next_target = target_lvl2[:, self.observed_claim_k]
                next_valid = target_mask_lvl2[:, self.observed_claim_k]
            else:
                next_target = target_lvl2
                next_valid = target_mask_lvl2
            per_hypothesis_mse = (
                multi_hypothesis_predictions - next_target.unsqueeze(1)
            ).pow(2).mean(dim=-1)
            per_hypothesis_mse = per_hypothesis_mse[next_valid]
            if per_hypothesis_mse.numel() > 0:
                temperature = self.multi_hypothesis_temperature
                softmin = -temperature * (
                    torch.logsumexp(-per_hypothesis_mse / temperature, dim=1)
                    - math.log(per_hypothesis_mse.size(1))
                )
                multi_hypothesis_loss = softmin.mean()
                multi_hypothesis_oracle_mse = per_hypothesis_mse.min(dim=1).values.mean()
        claim_prototype_loss = zero
        claim_prototype_prediction_loss = zero
        claim_prototype_clustering_loss = zero
        claim_prototype_top1_accuracy = zero
        claim_prototype_top5_accuracy = zero
        claim_prototype_target_entropy = zero
        claim_prototype_effective_count = zero
        claim_prototype_hard_utilization = zero
        claim_prototype_assignments = None
        claim_prototype_probabilities = None
        claim_prototype_active_mask = None
        if self.use_claim_prototypes:
            prototype_metrics = self.claim_prototype_objective(
                prediction_lvl2,
                target_lvl2,
                mask=target_mask_lvl2,
                sample_weights=lvl2_slot_weights,
            )
            claim_prototype_loss = prototype_metrics["total"]
            claim_prototype_prediction_loss = prototype_metrics["prediction"]
            claim_prototype_clustering_loss = prototype_metrics["clustering"]
            claim_prototype_top1_accuracy = prototype_metrics["top1_accuracy"]
            claim_prototype_top5_accuracy = prototype_metrics["top5_accuracy"]
            claim_prototype_target_entropy = prototype_metrics["target_entropy"]
            claim_prototype_effective_count = prototype_metrics["effective_prototypes"]
            claim_prototype_hard_utilization = prototype_metrics["hard_utilization"]
            claim_prototype_assignments = prototype_metrics["assignments"]
            claim_prototype_probabilities = prototype_metrics["student_probabilities"]
            claim_prototype_active_mask = prototype_metrics["active_mask"]
            # Keep the auxiliary under the existing Level-2 precision term so
            # enabling it does not add a new learnable log-var or change the
            # number of homoscedastic objectives in the anchor recipe.
            ssl_loss_lvl2 = (
                ssl_loss_lvl2
                + self.claim_prototype_weight * claim_prototype_loss
            )
        dense_observed_loss, dense_next_loss, dense_future_loss = self._compute_claim_slot_mse(
            prediction_lvl2,
            target_lvl2,
            target_mask_lvl2,
        )
        world_model_loss = zero
        world_model_next_loss = zero
        world_model_summary_loss = zero
        world_model_rollout = None
        world_model_state = None
        if self.use_world_model_dynamics:
            context_claim_mask = context_ttnc.ne(0)
            world_model_outputs = self.world_model(context_lvl2, context_claim_mask)
            world_model_targets = self._build_world_model_targets(target_lvl2, target_mask_lvl2)
            world_model_loss, world_model_next_loss, world_model_summary_loss = (
                self._compute_world_model_loss(world_model_outputs, world_model_targets)
            )
            world_model_rollout = world_model_outputs["world_model_rollout_claims"]
            world_model_state = world_model_outputs["world_model_state"]
        temporal_contrastive_loss = zero
        temporal_context_view = None
        temporal_future_view = None
        if self.use_temporal_contrastive:
            temporal_metrics = self._compute_temporal_contrastive_loss(
                sequence_aux["sequence_output"],
                context_ttnc,
                target_lvl2,
                target_mask_lvl2,
            )
            temporal_contrastive_loss = temporal_metrics["loss"]
            temporal_context_view = temporal_metrics["context_view"]
            temporal_future_view = temporal_metrics["future_view"]

        task_loss = zero
        predictor_head_input = None

        token_pred_loss = zero
        masked_token_grounding_loss = zero
        masked_next_claim_cpt_loss = zero
        masked_next_claim_icd_loss = zero
        masked_next_claim_ttnc_loss = zero
        masked_next_claim_cpt_mask_fraction = zero
        masked_next_claim_icd_mask_fraction = zero
        masked_next_claim_ttnc_mask_fraction = zero
        cpt_logits = None
        icd_logits = None
        ttnc_logits = None
        masked_next_claim_cpt_logits = None
        masked_next_claim_icd_logits = None
        masked_next_claim_ttnc_logits = None
        masked_target_cpt_tokens = None
        masked_target_icd_tokens = None
        if self.use_masked_next_claim_token_grounding and teacher_forcing:
            grounding_metrics = self._compute_masked_next_claim_token_grounding_loss(
                next_claim_latent,
                target_cpt.squeeze(1),
                target_icd.squeeze(1),
                target_ttnc.squeeze(1),
            )
            masked_token_grounding_loss = grounding_metrics["total"]
            masked_next_claim_cpt_loss = grounding_metrics["cpt_loss"]
            masked_next_claim_icd_loss = grounding_metrics["icd_loss"]
            masked_next_claim_ttnc_loss = grounding_metrics["ttnc_loss"]
            masked_next_claim_cpt_mask_fraction = grounding_metrics["cpt_mask_fraction"]
            masked_next_claim_icd_mask_fraction = grounding_metrics["icd_mask_fraction"]
            masked_next_claim_ttnc_mask_fraction = grounding_metrics["ttnc_mask_fraction"]
            masked_next_claim_cpt_logits = grounding_metrics["cpt_logits"]
            masked_next_claim_icd_logits = grounding_metrics["icd_logits"]
            masked_next_claim_ttnc_logits = grounding_metrics["ttnc_logits"]
            masked_target_cpt_tokens = grounding_metrics["target_cpt"]
            masked_target_icd_tokens = grounding_metrics["target_icd"]

        if self.use_token_prediction_head and teacher_forcing:
            initial_cpt = target_cpt.squeeze(1)[:, 0]
            cpt_embeds = self._get_active_code_embeddings()["cpt"](initial_cpt)
            logit_context = logit_context + cpt_embeds

            cpt_logits, icd_logits, ttnc_logits = self.logits_generator(logit_context)
            target_cpt_multi_hot = self.create_multi_hot_targets(
                target_cpt, self.cpt_vocab_size, padding_idx=0
            )
            target_icd_multi_hot = self.create_multi_hot_targets(
                target_icd, self.icd_vocab_size, padding_idx=0
            )

            criterion_bce = nn.BCEWithLogitsLoss()
            criterion_ce = nn.CrossEntropyLoss()
            cpt_loss = criterion_bce(cpt_logits, target_cpt_multi_hot)
            icd_loss = criterion_bce(icd_logits, target_icd_multi_hot)
            ttnc_loss = criterion_ce(ttnc_logits, target_ttnc.squeeze(1))
            token_pred_loss = cpt_loss + icd_loss + ttnc_loss

        sae_loss = zero
        gating_weight_mean = zero
        gating_sae_fraction = zero
        if self.use_sparse_autoencoder:
            recon = self.sparse_autoencoder(patient_representation)
            sae_loss = F.mse_loss(recon, patient_representation)
            if self.use_gated_fusion:
                sae_encoded = self.sparse_autoencoder.encoder(patient_representation)
                sae_embed = self.sae_to_embed(sae_encoded)
                gate_input = torch.cat([patient_representation, sae_embed], dim=-1)
                gate = self.gating_network(gate_input)
                sae_part = gate * sae_embed
                patient_part = (1 - gate) * patient_representation
                patient_representation = sae_part + patient_part
                gating_weight_mean = gate.mean()
                sae_contrib = torch.norm(sae_part, dim=-1).mean()
                patient_contrib = torch.norm(patient_part, dim=-1).mean()
                gating_sae_fraction = sae_contrib / (sae_contrib + patient_contrib + 1e-8)
                if self.debug_generation:
                    print(f"gating_sae_fraction={gating_sae_fraction.item():.3f}")


                # ─── Diffusion loss for the last-claim reconstruction ─────────
        predictor_head_input = self._select_predictor_head_input(
            sequence_aux,
            patient_representation_pre_sae,
            patient_representation,
        )
        if self.use_predictor_head and target is not None:
            target_mean = target.mean()
            target_std = target.std() + 1e-8  # Add epsilon to prevent division by zero
            target_normalized = (target - target_mean) / target_std
            target_pred = self.non_linear_predictor(predictor_head_input)
            target_normalized = target_normalized.unsqueeze(1)
            task_loss = self.loss_fn(target_pred, target_normalized)

        diffusion_loss = zero
        if self.use_diffusion and self.diffusion_weight > 0:
            # Tokens from the last claim in the batch. ``target_*`` has shape
            # ``[batch, 1, num_tokens]`` so we remove the claim dimension but
            # retain the per-code dimension for CPT/ICD. TTNC is a single code
            # per claim so it becomes ``[batch]``.
            cpt_last = target_cpt.squeeze(1)  # [batch, max_cpt_tokens]
            icd_last = target_icd.squeeze(1)  # [batch, max_icd_tokens]
            ttnc_last = target_ttnc.squeeze(1)  # [batch]

            diffusion_loss = self.diffusion_model.forward(
                cpt_last,
                icd_last,
                ttnc_last,
                condition=self._extract_next_claim_prediction(prediction_lvl2),
            )

        # Compute total loss
        total_loss, task_loss = self.calculate_total_loss(
            ssl_loss_lvl1,
            ssl_loss_lvl2,
            task_loss,
            lvl2_weight=self.level_2_weight,
            token_pred_loss=token_pred_loss,
            world_model_loss=world_model_loss,
            temporal_contrastive_loss=temporal_contrastive_loss,
            masked_token_grounding_loss=masked_token_grounding_loss,
            masked_claim_jepa_loss=masked_claim_jepa_loss,
            multi_hypothesis_loss=multi_hypothesis_loss,
            sae_loss=sae_loss,
            diffusion_loss=diffusion_loss,
        )

        return {
            'loss': total_loss,
            'ssl_loss_lvl1': ssl_loss_lvl1,
            'level1_predictive_objective_raw': level1_predictive_objective_raw,
            'ssl_loss_lvl2': ssl_loss_lvl2,
            'ssl_predictive_lvl1': ssl_predictive_lvl1,
            'ssl_predictive_lvl2': ssl_predictive_lvl2,
            'ssl_regularizer_lvl1': ssl_regularizer_lvl1,
            'ssl_regularizer_lvl2': ssl_regularizer_lvl2,
            'ssl_covariance_lvl1': cov_loss_lvl1,
            'ssl_covariance_lvl2': cov_loss_lvl2,
            'ssl_sigreg_raw_lvl1': sigreg_raw_lvl1,
            'ssl_sigreg_raw_lvl2': sigreg_raw_lvl2,
            'levjepa_patient_view_loss': levjepa_patient_view_loss,
            'levjepa_invariance_loss': levjepa_invariance_loss,
            'levjepa_sigreg_raw': levjepa_sigreg_raw,
            'levjepa_retained_claim_fraction': levjepa_retained_claim_fraction,
            'levjepa_global_projection': levjepa_global_projection,
            'levjepa_local_projections': levjepa_local_projections,
            'claim_prototype_loss': claim_prototype_loss,
            'claim_prototype_prediction_loss': claim_prototype_prediction_loss,
            'claim_prototype_clustering_loss': claim_prototype_clustering_loss,
            'claim_prototype_top1_accuracy': claim_prototype_top1_accuracy,
            'claim_prototype_top5_accuracy': claim_prototype_top5_accuracy,
            'claim_prototype_target_entropy': claim_prototype_target_entropy,
            'claim_prototype_effective_count': claim_prototype_effective_count,
            'claim_prototype_hard_utilization': claim_prototype_hard_utilization,
            'claim_prototype_assignments': claim_prototype_assignments,
            'claim_prototype_probabilities': claim_prototype_probabilities,
            'claim_prototype_active_mask': claim_prototype_active_mask,
            'level1_marginal_regularizer': level1_marginal_raw,
            'level1_marginal_loss': level1_marginal_loss,
            'level1_marginal_cpt': level1_marginal_by_modality['cpt'],
            'level1_marginal_icd': level1_marginal_by_modality['icd'],
            'world_model_loss': world_model_loss,
            'world_model_next_loss': world_model_next_loss,
            'world_model_summary_loss': world_model_summary_loss,
            'world_model_rollout': world_model_rollout,
            'world_model_state': world_model_state,
            'temporal_contrastive_loss': temporal_contrastive_loss,
            'temporal_context_view': temporal_context_view,
            'temporal_future_view': temporal_future_view,
            'var_loss_lvl1': var_loss_lvl1,
            'var_loss_lvl2': var_loss_lvl2,
            'inv_loss_lvl1': ssl_predictive_lvl1,
            'inv_loss_lvl2': ssl_predictive_lvl2,
            'vicreg_loss_lvl1': ssl_loss_lvl1,
            'vicreg_loss_lvl2': ssl_loss_lvl2,
            'var_pred_lvl1': embedding_variance_lvl1,
            'var_pred_lvl2': embedding_variance_lvl2,
            'dense_observed_loss': dense_observed_loss,
            'dense_next_loss': dense_next_loss,
            'dense_future_loss': dense_future_loss,
            'dense_target_mask': target_mask_lvl2,
            'patient_representation_pre_sae': patient_representation_pre_sae,
            'patient_representation': patient_representation,
            'predictor_head_input': predictor_head_input,
            'prediction_lvl2': prediction_lvl2,
            'target_lvl2': target_lvl2,
            'sequence_aux': sequence_aux,
            'logit_context': logit_context,
            'task_loss': task_loss,
            'task_loss_scaled': task_loss * self.task_loss_weight,
            'token_pred_loss': token_pred_loss,
            'masked_next_claim_token_loss': masked_token_grounding_loss,
            'masked_claim_jepa_loss': masked_claim_jepa_loss,
            'masked_claim_prediction': masked_claim_prediction,
            'masked_claim_target': masked_claim_target,
            'masked_claim_positions': masked_claim_positions,
            'masked_claim_active': masked_claim_active,
            'multi_hypothesis_loss': multi_hypothesis_loss,
            'multi_hypothesis_oracle_mse': multi_hypothesis_oracle_mse,
            'multi_hypothesis_predictions': multi_hypothesis_predictions,
            'masked_next_claim_cpt_loss': masked_next_claim_cpt_loss,
            'masked_next_claim_icd_loss': masked_next_claim_icd_loss,
            'masked_next_claim_ttnc_loss': masked_next_claim_ttnc_loss,
            'masked_next_claim_cpt_mask_fraction': masked_next_claim_cpt_mask_fraction,
            'masked_next_claim_icd_mask_fraction': masked_next_claim_icd_mask_fraction,
            'masked_next_claim_ttnc_mask_fraction': masked_next_claim_ttnc_mask_fraction,
            'sae_loss': sae_loss,
            'gating_weight_mean': gating_weight_mean,
            'gating_sae_fraction': gating_sae_fraction,
            'diffusion_loss': diffusion_loss,
            'cpt_logits': cpt_logits if self.use_token_prediction_head else None,
            'icd_logits': icd_logits if self.use_token_prediction_head else None,
            'ttnc_logits': ttnc_logits if self.use_token_prediction_head else None,
            'masked_next_claim_cpt_logits': masked_next_claim_cpt_logits,
            'masked_next_claim_icd_logits': masked_next_claim_icd_logits,
            'masked_next_claim_ttnc_logits': masked_next_claim_ttnc_logits,
            'masked_target_cpt_tokens': masked_target_cpt_tokens,
            'masked_target_icd_tokens': masked_target_icd_tokens,
        }

    def autoregressive_generation(self, cpt_tensor, icd_tensor, ttnc_tensor):
        batch_size = cpt_tensor.size(0)

        if self.use_diffusion:
            # Condition sampling on the predicted next-claim representation
            context_lvl2 = self.encode_claims(cpt_tensor, icd_tensor, ttnc_tensor)
            _, prediction_lvl2 = self.prediction_block_lvl2(context_lvl2, ttnc_tensor)
            logit_context = self._extract_next_claim_prediction(prediction_lvl2)
            outputs = self.diffusion_model.sample(
                batch_size,
                condition=logit_context,
            )
            if isinstance(outputs, dict):
                return {
                    'predicted_cpt_codes': outputs['cpt_tokens'],
                    'predicted_icd_codes': outputs['icd_tokens'],
                    'predicted_ttnc_code': outputs['ttnc_token'],
                    'cpt_entropy': outputs['cpt_entropy'],
                    'cpt_threshold': outputs['cpt_threshold'],
                    'icd_entropy': outputs['icd_entropy'],
                    'icd_threshold': outputs['icd_threshold'],
                }
            else:
                cpt_tokens, icd_tokens, ttnc_token = outputs
                return {
                    'predicted_cpt_codes': cpt_tokens,
                    'predicted_icd_codes': icd_tokens,
                    'predicted_ttnc_code': ttnc_token,
                }

        # Obtain initial patient representation
        context_lvl2 = self.encode_claims(cpt_tensor, icd_tensor, ttnc_tensor)
        _, prediction_lvl2 = self.prediction_block_lvl2(context_lvl2, ttnc_tensor)
        logit_context = self._extract_next_claim_prediction(prediction_lvl2)

        # === First Pass ===
        # Sample initial CPT code using multinomial sampling
        cpt_logits_initial, _, ttnc_logits = self.logits_generator(logit_context)
        cpt_probs_initial = torch.softmax(cpt_logits_initial, dim=-1)
        initial_cpt = torch.multinomial(cpt_probs_initial, num_samples=1).squeeze(1)  # [batch_size]

        # Update context with initial CPT embedding
        cpt_embeds = self._get_active_code_embeddings()["cpt"](initial_cpt)
        logit_context = logit_context + cpt_embeds

        # Select TTNC code using argmax
        ttnc_sampled = torch.argmax(ttnc_logits, dim=1)
        generated_ttnc = ttnc_sampled.long()

        # === Generate Remaining Codes ===
        # Generate logits for remaining CPT and ICD codes
        cpt_logits, icd_logits, _ = self.logits_generator(logit_context)

        # Apply sigmoid to get probabilities
        cpt_probs = torch.sigmoid(cpt_logits)
        icd_probs = torch.sigmoid(icd_logits)

        # Calculate normalized entropy and derive per-example thresholds
        cpt_entropy = calculate_entropy(cpt_probs)
        icd_entropy = calculate_entropy(icd_probs)
        cpt_norm_entropy = cpt_entropy / math.log(cpt_probs.size(-1))
        icd_norm_entropy = icd_entropy / math.log(icd_probs.size(-1))

        if self.debug_low_threshold:
            dynamic_cpt_threshold = self.threshold
            dynamic_icd_threshold = self.threshold
        else:
            shrink_cpt = 1.0 - cpt_norm_entropy
            shrink_icd = 1.0 - icd_norm_entropy
            dynamic_cpt_threshold = self.cpt_base_threshold * shrink_cpt
            dynamic_icd_threshold = self.icd_base_threshold * shrink_icd
            min_thresh = 0.1
            dynamic_cpt_threshold = dynamic_cpt_threshold.clamp(min=min_thresh)
            dynamic_icd_threshold = dynamic_icd_threshold.clamp(min=min_thresh)

        # Apply dynamic threshold to select codes
        cpt_predicted = (cpt_probs > dynamic_cpt_threshold.unsqueeze(-1)).long()
        # Add initial cpt
        cpt_predicted.scatter_(1, initial_cpt.unsqueeze(1), 1)
        icd_predicted = (icd_probs > dynamic_icd_threshold.unsqueeze(-1)).long()

        # Reimplement this (?) when other gradient breaking operations are fixed.
        # alpha = 100  # Adjust as needed
        # cpt_predicted = torch.sigmoid((cpt_probs - dynamic_cpt_threshold.unsqueeze(-1)) * alpha)
        # icd_predicted = torch.sigmoid((icd_probs - dynamic_icd_threshold.unsqueeze(-1)) * alpha)

        # Store generated codes
        generated_cpt = cpt_predicted
        generated_icd = icd_predicted

        return {
            'predicted_cpt_codes': generated_cpt,
            'predicted_icd_codes': generated_icd,
            'predicted_ttnc_code': generated_ttnc,
            'cpt_entropy': cpt_entropy,
            'cpt_threshold': dynamic_cpt_threshold,
            'icd_entropy': icd_entropy,
            'icd_threshold': dynamic_icd_threshold,
        }

    def training_step(self, batch, batch_idx):
        # Forward pass
        outputs = self.forward(
            cpt_tensor=batch[0],
            icd_tensor=batch[1],
            ttnc_tensor=batch[2],
            target=batch[3],
            teacher_forcing=True,
            generation=False
        )

        total_loss = outputs['loss']
        diffusion_loss = outputs['diffusion_loss']

        clamped_ssl_lvl2 = torch.clamp(self.log_vars['ssl_lvl2'], min=-5, max=5)
        precision_ssl_lvl2 = torch.exp(-clamped_ssl_lvl2)
        weighted_ssl_lvl2 = (
            outputs['ssl_loss_lvl2'] * precision_ssl_lvl2 * self.level_2_weight
        )
        clamped_world_model = torch.clamp(self.log_vars['world_model'], min=-5, max=5)
        precision_world_model = torch.exp(-clamped_world_model)
        weighted_world_model_loss = outputs['world_model_loss'] * precision_world_model
        clamped_temporal = torch.clamp(self.log_vars['temporal_contrastive'], min=-5, max=5)
        precision_temporal = torch.exp(-clamped_temporal)
        weighted_temporal_loss = (
            outputs['temporal_contrastive_loss'] * precision_temporal * self.temporal_loss_weight
        )
        clamped_task = torch.clamp(self.log_vars['task'], min=-5, max=5)
        precision_task = torch.exp(-clamped_task)
        weighted_task_loss = outputs['task_loss_scaled'] * precision_task
        if self.level_2_weight > 0:
            self.log(
                "ssl_lvl2",
                weighted_ssl_lvl2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "ssl_predictive_lvl2",
                outputs['ssl_predictive_lvl2'],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "ssl_regularizer_lvl2",
                outputs['ssl_regularizer_lvl2'],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.ssl_lvl2_raw_total += outputs['ssl_loss_lvl2'].detach().item()
            self.ssl_lvl2_wgt_total += weighted_ssl_lvl2.detach().item()
            self.ssl_batch_count += 1

            if self.use_level2_dense_prediction:
                self.log(
                    "dense_observed_loss",
                    outputs['dense_observed_loss'],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
                if self.future_claim_k > 0:
                    self.log(
                        "dense_future_loss",
                        outputs['dense_future_loss'],
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                    )
                self.log(
                    "dense_next_loss",
                    outputs['dense_next_loss'],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )

            if self.ssl_objective_type == "vicreg":
                self.log(
                    "vicreg_lvl2",
                    weighted_ssl_lvl2,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

            if self.use_levjepa_patient_views:
                self.log(
                    "levjepa_patient_view_loss",
                    outputs["levjepa_patient_view_loss"],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    "levjepa_invariance_loss",
                    outputs["levjepa_invariance_loss"],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    "levjepa_sigreg_raw",
                    outputs["levjepa_sigreg_raw"],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    "levjepa_retained_claim_fraction",
                    outputs["levjepa_retained_claim_fraction"],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )

            if self.use_claim_prototypes:
                for metric_name in (
                    "claim_prototype_loss",
                    "claim_prototype_prediction_loss",
                    "claim_prototype_clustering_loss",
                    "claim_prototype_top1_accuracy",
                    "claim_prototype_top5_accuracy",
                    "claim_prototype_target_entropy",
                    "claim_prototype_effective_count",
                    "claim_prototype_hard_utilization",
                ):
                    self.log(
                        metric_name,
                        outputs[metric_name],
                        on_step=False,
                        on_epoch=True,
                        logger=True,
                    )

        if self.use_world_model_dynamics:
            self.log(
                "world_model_loss",
                outputs["world_model_loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
            self.log(
                "world_model_loss_wgt",
                weighted_world_model_loss,
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "world_model_next_loss",
                outputs["world_model_next_loss"],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "world_model_summary_loss",
                outputs["world_model_summary_loss"],
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        if self.use_temporal_contrastive:
            self.log(
                "temporal_contrastive_loss",
                outputs["temporal_contrastive_loss"],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "temporal_contrastive_loss_wgt",
                weighted_temporal_loss,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        # --- Logging ---
        self.log('loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.use_diffusion and self.diffusion_weight > 0:
            self.log('diffusion_loss', diffusion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Preserve existing logging
        if self.use_token_prediction_head:
            self.log('token_pred_loss', outputs['token_pred_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.use_masked_next_claim_token_grounding:
            self.log(
                'masked_next_claim_token_loss',
                outputs['masked_next_claim_token_loss'],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                'masked_next_claim_cpt_loss',
                outputs['masked_next_claim_cpt_loss'],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                'masked_next_claim_icd_loss',
                outputs['masked_next_claim_icd_loss'],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                'masked_next_claim_cpt_mask_fraction',
                outputs['masked_next_claim_cpt_mask_fraction'],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                'masked_next_claim_icd_mask_fraction',
                outputs['masked_next_claim_icd_mask_fraction'],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            if self.masked_next_claim_include_ttnc:
                self.log(
                    'masked_next_claim_ttnc_loss',
                    outputs['masked_next_claim_ttnc_loss'],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
                self.log(
                    'masked_next_claim_ttnc_mask_fraction',
                    outputs['masked_next_claim_ttnc_mask_fraction'],
                    on_step=False,
                    on_epoch=True,
                    logger=True,
                )
        if self.use_masked_claim_jepa:
            self.log(
                "masked_claim_jepa_loss",
                outputs["masked_claim_jepa_loss"],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
        if self.use_multi_hypothesis_future:
            self.log(
                "multi_hypothesis_loss",
                outputs["multi_hypothesis_loss"],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "multi_hypothesis_oracle_mse",
                outputs["multi_hypothesis_oracle_mse"],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
        if self.use_sparse_autoencoder and self.sae_weight > 0:
            self.log('sae_loss', outputs['sae_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # Track SAE loss for epoch-level logging
            self.sae_loss_total += outputs['sae_loss'].item()
            self.sae_loss_count += 1
        
        if self.use_level1:
            self.log('ssl_lvl1', outputs['ssl_loss_lvl1'], on_step=False, on_epoch=True, logger=True)
            self.log('ssl_predictive_lvl1', outputs['ssl_predictive_lvl1'], on_step=False, on_epoch=True, logger=True)
            self.log('ssl_regularizer_lvl1', outputs['ssl_regularizer_lvl1'], on_step=False, on_epoch=True, logger=True)
            if self.use_composable_level1:
                self.log('level1_marginal_regularizer', outputs['level1_marginal_regularizer'], on_step=False, on_epoch=True, logger=True)
                self.log('level1_marginal_loss', outputs['level1_marginal_loss'], on_step=False, on_epoch=True, logger=True)
                self.log('level1_marginal_cpt', outputs['level1_marginal_cpt'], on_step=False, on_epoch=True, logger=True)
                self.log('level1_marginal_icd', outputs['level1_marginal_icd'], on_step=False, on_epoch=True, logger=True)
            if self.ssl_objective_type == "sigreg":
                self.log('ssl_sigreg_raw_lvl1', outputs['ssl_sigreg_raw_lvl1'], on_step=False, on_epoch=True, logger=True)
            if self.ssl_objective_type == "vicreg":
                self.log('Iloss1', outputs['inv_loss_lvl1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('Var1', outputs['var_pred_lvl1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.level_2_weight > 0:
            if self.ssl_objective_type == "sigreg":
                self.log('ssl_sigreg_raw_lvl2', outputs['ssl_sigreg_raw_lvl2'], on_step=False, on_epoch=True, logger=True)
            if self.ssl_objective_type == "vicreg":
                self.log('Iloss2', outputs['inv_loss_lvl2'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
                self.log('Var2', outputs['var_pred_lvl2'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.use_predictor_head:
            self.log('task_loss', outputs['task_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(
                'task_loss_scaled',
                outputs['task_loss_scaled'],
                on_step=False,
                on_epoch=True,
                logger=True,
            )
            self.log(
                'task_loss_wgt',
                weighted_task_loss,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

        # Dynamically print active losses for quick inspection
        active_losses = [("total", total_loss)]
        if self.level_2_weight > 0:
            active_losses.append(("ssl_lvl2", weighted_ssl_lvl2))
        if self.use_predictor_head and self.task_loss_weight > 0:
            active_losses.append(("task", weighted_task_loss))
        if self.use_world_model_dynamics:
            active_losses.append(("world_model", weighted_world_model_loss))
        if self.use_temporal_contrastive and self.temporal_loss_weight > 0:
            active_losses.append(("temporal_contrastive", weighted_temporal_loss))
        if self.use_sparse_autoencoder and self.sae_weight > 0:
            active_losses.append(("sae", outputs['sae_loss']))
        if self.use_token_prediction_head:
            active_losses.append(("token_pred", outputs['token_pred_loss']))
        if self.use_masked_next_claim_token_grounding:
            active_losses.append(("masked_token_grounding", outputs['masked_next_claim_token_loss']))
        if self.use_claim_prototypes:
            active_losses.append(("claim_prototypes", outputs['claim_prototype_loss']))
        if self.use_masked_claim_jepa:
            active_losses.append(("masked_claim_jepa", outputs['masked_claim_jepa_loss']))
        if self.use_multi_hypothesis_future:
            active_losses.append(("multi_hypothesis", outputs['multi_hypothesis_loss']))
        if self.use_diffusion and self.diffusion_weight > 0:
            active_losses.append(("diffusion", diffusion_loss))
        # Active losses are logged via PyTorch Lightning at epoch end

        with torch.no_grad():
            if self.use_token_prediction_head:
                gen_metrics = self.autoregressive_generation(
                    batch[0], batch[1], batch[2]
                )
                if isinstance(gen_metrics, dict) and 'cpt_entropy' in gen_metrics:
                    self.log(
                        "avg_cpt_entropy",
                        gen_metrics["cpt_entropy"].mean(),
                        on_epoch=True,
                    )
                    self.log(
                        "avg_cpt_threshold",
                        gen_metrics["cpt_threshold"].mean(),
                        on_epoch=True,
                    )
                    self.log(
                        "avg_icd_entropy",
                        gen_metrics["icd_entropy"].mean(),
                        on_epoch=True,
                    )
                    self.log(
                        "avg_icd_threshold",
                        gen_metrics["icd_threshold"].mean(),
                        on_epoch=True,
                    )

        # Update target encoders after each step
        self.update_target_encoders()
        
        # Accumulate representations and targets for regression evaluation
        self.repr_accumulator.append(outputs['patient_representation'].detach().cpu())
        self.target_accumulator.append(batch[3].detach().cpu())
        
        return total_loss  # Return the total loss for logging purposes


    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=None):
        self.update_eval_polyak_average()
        if (batch_idx + 1) == self.steps_per_epoch:
            if self.use_grad_print:
                total_norm = 0
                for name, param in self.named_parameters():
                    if param.grad is not None:
                        param_norm = param.grad.data.norm(2)
                        print(f"Grad Norm for {name}: {param_norm.item()}")
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                print(f"Total Gradient Norm: {total_norm}")

            self.prediction_block_lvl2.on_epoch_end()

    def on_validation_start(self):
        self.activate_eval_polyak_weights()

    def on_validation_epoch_end(self):
        # Restore before validation-end checkpoint callbacks can serialize the
        # model, keeping checkpoints authoritative for the online parameters.
        self.restore_online_weights()

    def on_validation_end(self):
        # Safety net for interrupted or non-standard evaluation loops.
        self.restore_online_weights()

    def on_after_backward(self):
        if not self._grad_check_done:
            for name, param in self.named_parameters():
                if "encoder_lvl" in name:
                    _ = param.grad
            self._grad_check_done = True

    def _maybe_freeze_logvars(self):
        freeze_epoch = self.freeze_logvars_after_epoch
        if freeze_epoch is None or self._logvars_frozen:
            return
        if self.current_epoch < freeze_epoch:
            return

        for param in self.log_vars.parameters():
            param.requires_grad_(False)
            if param.grad is not None:
                param.grad = None
        self._logvars_frozen = True

    def on_train_epoch_start(self):
        self._maybe_freeze_logvars()


    def train_linear_regression(self, X_sample, y_sample):
        """Train the linear model using sampled representations"""
        # Add a bias term if necessary
        ones = torch.ones(X_sample.size(0), 1, device=X_sample.device)
        X_sample = torch.cat([X_sample, ones], dim=1)
        X_sample = X_sample.to(self.device)
        y_sample = y_sample.to(self.device)

        # Compute least squares solution
        solution = torch.linalg.lstsq(X_sample, y_sample)
        self.regression_weights = solution.solution.to(self.device)

    def on_train_epoch_end(self):
        """Log average SAE loss at the end of each epoch."""
        if self.use_sparse_autoencoder and self.sae_weight > 0 and self.sae_loss_count > 0:
            avg_sae_loss = self.sae_loss_total / self.sae_loss_count
            self.log('avg_sae_loss', avg_sae_loss, prog_bar=True, logger=True)
            self.sae_loss_total = 0.0
            self.sae_loss_count = 0

        if self.ssl_batch_count > 0 and self.level_2_weight > 0:
            avg_raw = self.ssl_lvl2_raw_total / self.ssl_batch_count
            avg_wgt = self.ssl_lvl2_wgt_total / self.ssl_batch_count
            self.log('ssl_lvl2_raw', avg_raw, prog_bar=True, logger=True)
            self.log('ssl_lvl2_wgt', avg_wgt, prog_bar=True, logger=True)
            if self.ssl_objective_type == "vicreg":
                self.log('vicreg_lvl2_raw', avg_raw, prog_bar=True, logger=True)
                self.log('vicreg_lvl2_wgt', avg_wgt, prog_bar=True, logger=True)
            self.ssl_lvl2_raw_total = 0.0
            self.ssl_lvl2_wgt_total = 0.0
            self.ssl_batch_count = 0

        # ----- End of epoch cross-validation for regression -----
        if len(self.repr_accumulator) > 0:
            X_accum = torch.cat(self.repr_accumulator)
            y_accum = torch.cat(self.target_accumulator)

            if self.use_zero_target_mask:
                non_zero_mask = y_accum != 0
                X_accum = X_accum[non_zero_mask]
                y_accum = y_accum[non_zero_mask]

            X_np = X_accum.cpu().numpy()
            y_np = y_accum.cpu().numpy()

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            rmse_list = []
            for train_index, val_index in kf.split(X_np):
                X_train, X_val = X_np[train_index], X_np[val_index]
                y_train, y_val = y_np[train_index], y_np[val_index]

                scaler_X = StandardScaler()
                X_train_np = scaler_X.fit_transform(X_train)
                X_val_np = scaler_X.transform(X_val)
                scaler_y = StandardScaler()
                y_train_np = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                y_val_np = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

                X_train_torch = torch.from_numpy(X_train_np).float().to(self.device)
                y_train_torch = torch.from_numpy(y_train_np).float().to(self.device)
                X_val_torch = torch.from_numpy(X_val_np).float().to(self.device)
                y_val_torch = torch.from_numpy(y_val_np).float().to(self.device)

                self.train_linear_regression(X_train_torch, y_train_torch)
                val_rmse = calculate_rmse(
                    self.regression_weights, X_val_torch, y_val_torch, scaler_y
                )
                if val_rmse is not None:
                    rmse_list.append(val_rmse)

            if rmse_list:
                avg_rmse = sum(rmse_list) / len(rmse_list)
                self.log(
                    "train_epoch_cv_probe_rmse",
                    avg_rmse,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                baseline_rmse = self.mean_target_baseline_rmse
                if baseline_rmse is not None and math.isfinite(baseline_rmse) and baseline_rmse > 0:
                    improvement = baseline_rmse - avg_rmse
                    improvement_pct = 100.0 * improvement / baseline_rmse
                    # This remains a training-set diagnostic. Frozen-split
                    # evaluation is the model-selection authority.
                    self.log("mean_target_baseline_rmse", baseline_rmse, on_epoch=True, logger=True)
                    self.log(
                        "train_epoch_cv_probe_improvement_vs_mean_baseline",
                        improvement,
                        on_epoch=True,
                        logger=True,
                    )
                    self.log(
                        "train_epoch_cv_probe_improvement_pct_vs_mean_baseline",
                        improvement_pct,
                        on_epoch=True,
                        logger=True,
                    )

        active_logvars = {}
        if self.use_level1:
            active_logvars['ssl_lvl1'] = self.log_vars['ssl_lvl1']
        if self.level_2_weight > 0:
            active_logvars['ssl_lvl2'] = self.log_vars['ssl_lvl2']
        if self.use_world_model_dynamics:
            active_logvars['world_model'] = self.log_vars['world_model']
        if self.use_temporal_contrastive and self.temporal_loss_weight > 0:
            active_logvars['temporal_contrastive'] = self.log_vars['temporal_contrastive']
        if self.use_predictor_head:
            active_logvars['task'] = self.log_vars['task']
        if self.use_token_prediction_head:
            active_logvars['token_pred'] = self.log_vars['token_pred']
        if self.use_masked_claim_jepa and self.masked_claim_jepa_weight > 0:
            active_logvars['masked_claim_jepa'] = self.log_vars['masked_claim_jepa']
        if self.use_multi_hypothesis_future and self.multi_hypothesis_weight > 0:
            active_logvars['multi_hypothesis'] = self.log_vars['multi_hypothesis']
        if self.use_sparse_autoencoder and self.sae_weight > 0:
            active_logvars['sae'] = self.log_vars['sae']
        if self.use_diffusion and self.diffusion_weight > 0:
            active_logvars['diffusion'] = self.log_vars['diffusion']

        for name, param in active_logvars.items():
            raw_logvar = float(param.detach().item())
            clamped_logvar = float(torch.clamp(param.detach(), min=-5, max=5).item())
            precision = float(torch.exp(-torch.tensor(clamped_logvar)).item())
            self.log(f"logvar_{name}", raw_logvar, prog_bar=True, logger=True)
            self.log(
                f"clamped_logvar_{name}",
                clamped_logvar,
                prog_bar=False,
                logger=True,
            )
            self.log(
                f"precision_{name}",
                precision,
                prog_bar=False,
                logger=True,
            )
        self.log(
            "logvars_frozen",
            float(self._logvars_frozen),
            prog_bar=False,
            logger=True,
        )

        if self.use_diffusion and self.diffusion_weight > 0:
            clamped = torch.clamp(self.log_vars['diffusion'], min=-5, max=5)
            lr_mult = torch.exp(-clamped)
            self.log("diff_lr_mult", lr_mult.item(), prog_bar=True, logger=True)

        self.repr_accumulator.clear()
        self.target_accumulator.clear()

    def configure_optimizers(self):
        # Parameters divided into encoder adapters and generator modules
        adapter_params = []
        generator_params = []
        collected_parameter_ids = set()

        def collect_params(module, into_list):
            for p in module.parameters():
                if p.requires_grad and id(p) not in collected_parameter_ids:
                    into_list.append(p)
                    collected_parameter_ids.add(id(p))

        if self.use_level1:
            collect_params(self.context_encoder_lvl1, adapter_params)
            collect_params(self.target_encoder_lvl1, adapter_params)
            collect_params(self.prediction_block_lvl1, generator_params)
            if self.use_composable_level1:
                collect_params(self.context_level1_composer, adapter_params)
                collect_params(self.target_level1_composer, adapter_params)

        if not self.use_composable_level1:
            collect_params(self.context_encoder_lvl2, adapter_params)
            collect_params(self.target_encoder_lvl2, adapter_params)
        collect_params(self.prediction_block_lvl2, generator_params)
        if self.use_levjepa_patient_views:
            collect_params(self.levjepa_patient_projector, generator_params)
        if self.use_world_model_dynamics:
            collect_params(self.world_model, generator_params)
        if self.use_temporal_contrastive:
            collect_params(self.temporal_contrastive_head, generator_params)
        if self.use_claim_prototypes:
            collect_params(self.claim_prototype_objective, generator_params)
        if self.use_masked_claim_jepa:
            collect_params(self.masked_claim_query_head, generator_params)
        if self.use_multi_hypothesis_future:
            collect_params(self.multi_hypothesis_future_head, generator_params)

        if self.use_sparse_autoencoder:
            collect_params(self.sparse_autoencoder, generator_params)
            if self.use_gated_fusion:
                collect_params(self.sae_to_embed, generator_params)
                collect_params(self.gating_network, generator_params)

        collect_params(self.log_vars, generator_params)
        if self.use_token_prediction_head:
            collect_params(self.logits_generator, generator_params)
            generator_params.append(self.threshold)
            generator_params.append(self.lambda_entropy)

        if self.use_predictor_head:
            collect_params(self.non_linear_predictor, generator_params)

        optimizer_class = (
            torch.optim.AdamW
            if self.optimizer_type == "adamw"
            else torch.optim.Adam
        )
        optimizer_gen = optimizer_class(
            [
                {
                    'params': adapter_params,
                    'lr': self.adapter_lr,
                    'weight_decay': self.weight_decay,
                },
                {
                    'params': generator_params,
                    'lr': self.generator_lr,
                    'weight_decay': self.weight_decay,
                },
            ],
            betas=self.optimizer_betas,
        )

        scheduler = None
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer_gen,
                step_size=self.scheduler_step_size,
                gamma=self.scheduler_gamma,
            )
        elif self.scheduler_type == "cosine":
            t_max = self.scheduler_t_max
            if t_max is None:
                t_max = max(1, self.training_epochs - self.scheduler_warmup_epochs)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer_gen,
                T_max=t_max,
                eta_min=self.scheduler_eta_min,
            )

        if self.scheduler_warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer_gen,
                start_factor=self.scheduler_warmup_start_factor,
                end_factor=1.0,
                total_iters=self.scheduler_warmup_epochs,
            )
            if scheduler is None:
                scheduler = warmup
            else:
                scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer_gen,
                    schedulers=[warmup, scheduler],
                    milestones=[self.scheduler_warmup_epochs],
                )

        if scheduler is None:
            return [optimizer_gen]
        return [optimizer_gen], [scheduler]


