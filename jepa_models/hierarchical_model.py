# models/hierarchical_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import math
import warnings
from jepa_models.encoders import Level1Encoder, Level2Encoder
from jepa_models.prediction_blocks import Level1PredictionBlock, Level2PredictionBlock, LogitsGenerator
from diffusion_models import ClaimD3PM
from jepa_models.sparse_autoencoder import SparseAutoencoder
from jepa_utils.metrics import calculate_rmse
from jepa_utils.tensor_utils import calculate_entropy
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

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
    def __init__(self, config, diffusion=None):
        super(HierarchicalClaimsModel, self).__init__()
        self.automatic_optimization = True
        self.save_hyperparameters()
        self.config = config
        if getattr(config, 'debug_generation', False):
            print("Initializing HierarchicalClaimsModel")
            print(f"cpt_vocab_size: {config.cpt_vocab_size}")
            print(f"icd_vocab_size: {config.icd_vocab_size}")
            print(f"ttnc_vocab_size: {config.ttnc_vocab_size}")
            print(f"embedding_dim: {config.embedding_dim}")

        # Configuration parameters
        self.ema_decay = config.ema_decay
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
        self.max_generated_tokens = config.max_generated_tokens
        self.use_predictor_head = config.use_predictor_head
        self.use_grad_print = config.use_grad_print
        self.use_level1 = config.use_level1
        self.use_zero_target_mask = config.use_zero_target_mask
        self.use_token_prediction_head = config.use_token_prediction_head
        self.use_sparse_autoencoder = config.use_sparse_autoencoder
        self.use_gated_fusion = config.use_gated_fusion
        self.use_diffusion = getattr(config, 'use_diffusion', False)
        self.diffusion_weight = getattr(config, 'diffusion_weight', 1.0)
        self.warmup_logvar_epochs = getattr(config, 'warmup_logvar_epochs', 3)
        self.debug_generation = getattr(config, 'debug_generation', False)
        self.cpt_vocab_size = config.cpt_vocab_size
        self.icd_vocab_size = config.icd_vocab_size
        self.is_stage1_pretrain = getattr(config, 'current_stage', 'stage1') == 'stage1'

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
        self.vicreg_lvl2_raw_total = 0.0
        self.vicreg_lvl2_wgt_total = 0.0
        self.vicreg_batch_count = 0

        self.threshold = nn.Parameter(torch.tensor(0.1))
        self.lambda_entropy = nn.Parameter(torch.tensor(config.lambda_entropy))

        # Initialize Encoders and Prediction Blocks
        if self.use_level1:
            self.context_encoder_lvl1 = Level1Encoder(
                cpt_vocab_size=config.cpt_vocab_size,
                icd_vocab_size=config.icd_vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=0
            )
            self.target_encoder_lvl1 = Level1Encoder(
                cpt_vocab_size=config.cpt_vocab_size,
                icd_vocab_size=config.icd_vocab_size,
                embedding_dim=config.embedding_dim,
                padding_idx=0
            )
            self.prediction_block_lvl1 = Level1PredictionBlock(
                embedding_dim=config.embedding_dim
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
        )

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
        self.sae_weight = getattr(config, "sae_weight", 1.0)
        self.loss_fn = nn.MSELoss()
        self.log_vars = nn.ParameterDict({
            'vicreg_lvl1': nn.Parameter(torch.tensor(0.5)),
            'vicreg_lvl2': nn.Parameter(torch.tensor(0.5)),
            'task': nn.Parameter(torch.tensor(0.5)),
            'token_pred': nn.Parameter(torch.tensor(0.5)),
            'sae': nn.Parameter(torch.tensor(0.5)),
            'diffusion': nn.Parameter(torch.tensor(0.5)),
        })

        # Track how long each precision stays at a clamp edge
        self.edge_streaks = {k: 0 for k in self.log_vars.keys()}

        if self.use_predictor_head:
            self.non_linear_predictor = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),  # First layer
                nn.ReLU(),  # Non-linearity
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),  # Second layer
                nn.ReLU(),  # Non-linearity
                nn.Linear(config.hidden_dim // 2, 1)  # Output layer for regression
            )

        if self.use_token_prediction_head:
            self.logits_generator = LogitsGenerator(config)

        if diffusion is not None:
            self.diffusion_model = diffusion
            self.use_diffusion = True
        elif self.use_diffusion:
            vocab_size = (
                config.cpt_vocab_size
                + config.icd_vocab_size
                + config.ttnc_vocab_size
                + 3
            )
            self.diffusion_model = ClaimD3PM(
                config,
                vocab_size,
                condition_dim=config.output_dim,
            )

        self.initialize_target_encoders()

        if (
            getattr(config, "current_stage", "stage1") == "stage2"
            and getattr(config, "freeze_encoder_at_stage2", True)
        ):
            self.freeze_encoder(getattr(config, "encoder_unfreeze_layers", 1))
            if self.use_sparse_autoencoder:
                for p in self.sparse_autoencoder.parameters():
                    p.requires_grad = False

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

    def freeze_encoders(self, n_layers: int = 1):
        """Freeze all encoder weights except the last ``n_layers``."""
        modules = [self.context_encoder_lvl2, self.target_encoder_lvl2]
        if self.use_level1:
            modules.extend([self.context_encoder_lvl1, self.target_encoder_lvl1])
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
        for mod in modules:
            for param in mod.parameters():
                param.requires_grad = True

    def initialize_target_encoders(self):
        if self.use_level1:
            for param_q, param_k in zip(self.context_encoder_lvl1.parameters(), self.target_encoder_lvl1.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        # Copy context encoder parameters to target encoders and set requires_grad=False
        for param_q, param_k in zip(self.context_encoder_lvl2.parameters(), self.target_encoder_lvl2.parameters()):
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
        with torch.no_grad():
            if self.use_level1:
                for param_q, param_k in zip(self.context_encoder_lvl1.parameters(), self.target_encoder_lvl1.parameters()):
                    param_k.data = self.ema_decay * param_k.data + (1.0 - self.ema_decay) * param_q.data

            for param_q, param_k in zip(self.context_encoder_lvl2.parameters(), self.target_encoder_lvl2.parameters()):
                param_k.data = self.ema_decay * param_k.data + (1.0 - self.ema_decay) * param_q.data

    def calculate_vicreg_loss(self, context_output, target_output, level):
        if level == '1':
            target_var = self.target_var_lvl1
        else:
            target_var = self.target_var_lvl2
        # Center the outputs
        context_output = context_output - context_output.mean(dim=0, keepdim=True)
        target_output = target_output - target_output.mean(dim=0, keepdim=True)

        # Variance term
        std_context = torch.sqrt(context_output.var(dim=0) + self.epsilon)
        std_target = torch.sqrt(target_output.var(dim=0) + self.epsilon)
        var_loss = (torch.mean(F.relu(target_var - std_context)) / 2 + 
                    torch.mean(F.relu(target_var - std_target)) / 2)**self.amplification_power

        # Invariance term (MSE)
        inv_loss = self.loss_fn(context_output, target_output)

        # Covariance term
        cov_context = (context_output.T @ context_output) / (context_output.size(0) - 1)
        cov_target = (target_output.T @ target_output) / (target_output.size(0) - 1)
        I = torch.eye(cov_context.size(0)).to(context_output.device)
        cov_loss = ((cov_context - I) ** 2).sum() / 2 + ((cov_target - I) ** 2).sum() / 2

        # Combining losses based on level
        if level == '1':
            vicreg_loss = (var_loss * self.var_penalty_scale_lvl1 + inv_loss + cov_loss * self.cov_penalty_scale_lvl1) 
        else:
            vicreg_loss = (var_loss * self.var_penalty_scale_lvl2 + inv_loss + cov_loss * self.cov_penalty_scale_lvl2)

        return vicreg_loss, var_loss, inv_loss

    def calculate_total_loss(self, vicreg_loss_lvl1, vicreg_loss_lvl2, task_loss, lvl2_weight, token_pred_loss, sae_loss=0, diffusion_loss=0):
        # Conceptually - https://arxiv.org/pdf/1705.07115
        log_var_denom = 2
        log_var_denom += 1 if self.use_level1 else 0
        log_var_denom += 1 if self.use_predictor_head else 0
        log_var_denom += 1 if self.use_token_prediction_head else 0
        log_var_denom += 1 if self.use_sparse_autoencoder else 0
        log_var_denom += 1 if self.use_diffusion else 0


        # Clamp precision log-variance to keep scaling factors stable
        clamped_log_vars = {k: torch.clamp(v, min=-3, max=1) for k, v in self.log_vars.items()}
        precision_vicreg_lvl2 = torch.exp(-clamped_log_vars['vicreg_lvl2'])

        if self.use_level1:
            precision_vicreg_lvl1 = torch.exp(-clamped_log_vars['vicreg_lvl1'])
            weighted_vicreg_lvl1_loss = vicreg_loss_lvl1 * precision_vicreg_lvl1
            total_loss = weighted_vicreg_lvl1_loss
            total_precision = precision_vicreg_lvl1
            total_log_var = clamped_log_vars['vicreg_lvl1']
        else:
            total_loss = torch.tensor(0.0, device=self.device)
            total_precision = 0
            total_log_var = 0

        weighted_vicreg_lvl2_loss = vicreg_loss_lvl2 * precision_vicreg_lvl2 * lvl2_weight
        total_loss = total_loss + weighted_vicreg_lvl2_loss
        total_precision = total_precision + precision_vicreg_lvl2
        total_log_var = total_log_var + clamped_log_vars['vicreg_lvl2']

        if self.use_predictor_head:
            precision_task = torch.exp(-clamped_log_vars['task'])
            weighted_task_loss = task_loss * precision_task
            total_loss = total_loss + weighted_task_loss
            total_precision = total_precision + precision_task
            total_log_var = total_log_var + clamped_log_vars['task']

        if self.use_token_prediction_head:
            precision_token = torch.exp(-clamped_log_vars['token_pred'])
            weighted_token_loss = token_pred_loss * precision_token
            total_loss = total_loss + weighted_token_loss
            total_precision = total_precision + precision_token
            total_log_var = total_log_var + clamped_log_vars['token_pred']

        if self.use_sparse_autoencoder:
            precision_sae = torch.exp(-clamped_log_vars['sae'])
            weighted_sae_loss = sae_loss * precision_sae * self.sae_weight
            total_loss = total_loss + weighted_sae_loss
            total_precision = total_precision + precision_sae
            total_log_var = total_log_var + clamped_log_vars['sae']

        if self.use_diffusion:
            precision_diff = torch.exp(-clamped_log_vars['diffusion'])
            weighted_diff_loss = diffusion_loss * precision_diff * self.diffusion_weight
            total_loss = total_loss + weighted_diff_loss
            total_precision = total_precision + precision_diff
            total_log_var = total_log_var + clamped_log_vars['diffusion']


        # Averaging the total loss and adding regularization
        total_loss = total_loss / (total_precision + 1e-8)
        total_loss = total_loss + total_log_var / log_var_denom

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

    def forward(self, cpt_tensor, icd_tensor, ttnc_tensor, target=None, teacher_forcing=True, generation=False):
        if generation:
            return self.autoregressive_generation(cpt_tensor, icd_tensor, ttnc_tensor)
        else:
            return self.training_forward(cpt_tensor, icd_tensor, ttnc_tensor, target, teacher_forcing)

    def training_forward(self, cpt_tensor, icd_tensor, ttnc_tensor, target, teacher_forcing=True):
        # Handle Level 1 Encoding and Prediction
        if self.use_level1:
            # Level 1: Within-claims prediction
            # --- First Pair: CPT as context, ICD as target ---
            context_lvl1_cpt, mask_cpt = self.context_encoder_lvl1(cpt_tensor, 'cpt')
            target_lvl1_icd, mask_icd_target = self.target_encoder_lvl1(icd_tensor, 'icd')

            prediction_lvl1_cpt = self.prediction_block_lvl1(context_lvl1_cpt, mask_cpt)

            batch_size, num_claims, total_embedding_dim = prediction_lvl1_cpt.shape
            embedding_dim = total_embedding_dim // 3

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
            vicreg_loss_lvl1_cpt, var_loss_lvl1_cpt, inv_loss_lvl1_cpt = self.calculate_vicreg_loss(
                prediction_lvl1_cpt_flat, target_lvl1_icd_flat, "1"
            )

            # Second Pair: ICD as context, CPT as target
            # Pass icd_tensor to context_encoder_lvl1 and cpt_tensor to target_encoder_lvl1
            context_lvl1_icd, mask_icd = self.context_encoder_lvl1(icd_tensor, 'icd')
            target_lvl1_cpt, mask_cpt_target = self.target_encoder_lvl1(cpt_tensor, 'cpt')

            prediction_lvl1_icd = self.prediction_block_lvl1(context_lvl1_icd, mask_icd)

            batch_size, num_claims, total_embedding_dim = prediction_lvl1_icd.shape
            embedding_dim = total_embedding_dim // 3

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
            vicreg_loss_lvl1_icd, var_loss_lvl1_icd, inv_loss_lvl1_icd = self.calculate_vicreg_loss(
                prediction_lvl1_icd_flat, target_lvl1_cpt_flat, "1"
            )

            # Combine losses
            vicreg_loss_lvl1 = (vicreg_loss_lvl1_cpt + vicreg_loss_lvl1_icd) / 2
            var_loss_lvl1 = (var_loss_lvl1_cpt + var_loss_lvl1_icd) / 2
            inv_loss_lvl1 = (inv_loss_lvl1_cpt + inv_loss_lvl1_icd) / 2

            # Embedding variance
            embedding_variance_lvl1 = (
                torch.var(prediction_lvl1_cpt_flat, dim=0).mean() +
                torch.var(prediction_lvl1_icd_flat, dim=0).mean()
            ) / 2
        else:
            # Handle case when Level 1 is not used
            vicreg_loss_lvl1 = torch.tensor(0.0, device=self.device)
            var_loss_lvl1 = torch.tensor(0.0, device=self.device)
            inv_loss_lvl1 = torch.tensor(0.0, device=self.device)
            embedding_variance_lvl1 = torch.tensor(0.0, device=self.device)

        # Level 2: Sequence to predict next claim representation
        # Split the context and target tensors
        context_cpt = cpt_tensor[:, :-1]
        context_icd = icd_tensor[:, :-1]
        context_ttnc = ttnc_tensor[:, :-1]

        # Select only the last claim for the target
        target_cpt = cpt_tensor[:, -1].unsqueeze(1)  # Adding a dimension to keep shape consistent
        target_icd = icd_tensor[:, -1].unsqueeze(1)
        target_ttnc = ttnc_tensor[:, -1].unsqueeze(1)

        # Level 2: Across-claims prediction
        context_lvl2 = self.context_encoder_lvl2(context_cpt, context_icd, context_ttnc)
        target_lvl2 = self.target_encoder_lvl2(target_cpt, target_icd, target_ttnc).squeeze(1)
        # Target encoding (real claim representations)
        real_target_lvl2 = self.target_encoder_lvl2(target_cpt, target_icd, target_ttnc).squeeze(1)

        patient_representation, prediction_lvl2 = self.prediction_block_lvl2(
                context_lvl2, context_ttnc
            )

        # --- Quick-Look Debug Artifact --------------------------------------
        if self.current_epoch == 0 and self.global_step == 0:
            claim_valid_mask = context_ttnc != 0
            valid_counts = claim_valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            mean_abs_embed = patient_representation.abs().mean()

        context_padding_mask = (context_ttnc != 0).float()
        target_padding_mask = (target_ttnc != 0).float()

        # A sequence is valid if it has at least one non-padding token in both context and target
        valid_sequences_mask = (context_padding_mask.sum(dim=1) > 0) & (target_padding_mask.sum(dim=1) > 0)

        # Compute variance across the valid sequences
        embedding_variance_lvl2 = torch.var(prediction_lvl2[valid_sequences_mask], dim=0).mean()

        vicreg_loss_lvl2, var_loss_lvl2, inv_loss_lvl2 = self.calculate_vicreg_loss(
            prediction_lvl2[valid_sequences_mask],
            target_lvl2[valid_sequences_mask],
            "2"
        )

        task_loss = 0
        if self.use_predictor_head and target is not None:
            target_mean = target.mean()
            target_std = target.std() + 1e-8  # Add epsilon to prevent division by zero
            # Normalize the target
            target_normalized = (target - target_mean) / target_std

            target_pred = self.non_linear_predictor(context_lvl2.mean(dim=1))  # Shape [batch_size, 1]

            target_normalized = target_normalized.unsqueeze(1)  # Make sure target is [batch_size, 1]

            task_loss = self.loss_fn(target_pred, target_normalized)

        token_pred_loss = 0  # Initialize token prediction loss
        logit_context = prediction_lvl2
        if self.use_token_prediction_head:
            batch_size = cpt_tensor.size(0)
            if teacher_forcing:
                # Use ground truth initial CPT code
                initial_cpt = target_cpt.squeeze(1)[:, 0]  # First CPT code

                # Update context with initial CPT embedding
                cpt_embeds = self.target_encoder_lvl2.cpt_embedding(initial_cpt)
                logit_context = logit_context + cpt_embeds.squeeze(1)

                # Use ground truth TTNC code
                ttnc_sampled = target_ttnc.squeeze(1)
                generated_ttnc = ttnc_sampled

                # Generate logits for CPT, ICD, and TTNC codes
                cpt_logits, icd_logits, ttnc_logits = self.logits_generator(logit_context)

                # Create multi-hot target vectors
                target_cpt_multi_hot = self.create_multi_hot_targets(target_cpt, self.cpt_vocab_size, padding_idx=0)
                target_icd_multi_hot = self.create_multi_hot_targets(target_icd, self.icd_vocab_size, padding_idx=0)

                # Compute loss using targets
                criterion_bce = nn.BCEWithLogitsLoss()
                criterion_ce = nn.CrossEntropyLoss()
                cpt_loss = criterion_bce(cpt_logits, target_cpt_multi_hot)
                icd_loss = criterion_bce(icd_logits, target_icd_multi_hot)
                ttnc_loss = criterion_ce(ttnc_logits, target_ttnc.squeeze(1))
                token_pred_loss = cpt_loss + icd_loss + ttnc_loss

        sae_loss = 0
        gating_weight_mean = torch.tensor(0.0, device=self.device)
        gating_sae_fraction = torch.tensor(0.0, device=self.device)
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
        diffusion_loss = torch.tensor(0.0, device=self.device)          # default no-op
        if self.use_diffusion:
            # Tokens from the last claim in the batch. ``target_*`` has shape
            # ``[batch, 1, num_tokens]`` so we remove the claim dimension but
            # retain the per-code dimension for CPT/ICD. TTNC is a single code
            # per claim so it becomes ``[batch]``.
            cpt_last = target_cpt.squeeze(1)  # [batch, max_cpt_tokens]
            icd_last = target_icd.squeeze(1)  # [batch, max_icd_tokens]
            ttnc_last = target_ttnc.squeeze(1)  # [batch]

            tokens = torch.cat([cpt_last, icd_last, ttnc_last.unsqueeze(1)], dim=1)
            diffusion_loss = self.diffusion_model.forward(
                tokens,
                condition=prediction_lvl2,
            )

        # Compute total loss
        total_loss, task_loss = self.calculate_total_loss(
            vicreg_loss_lvl1,
            vicreg_loss_lvl2,
            task_loss,
            lvl2_weight=self.level_2_weight,
            token_pred_loss=token_pred_loss,
            sae_loss=sae_loss,
            diffusion_loss=diffusion_loss,
        )

        return {
            'loss': total_loss,
            'var_loss_lvl1': var_loss_lvl1,
            'var_loss_lvl2': var_loss_lvl2,
            'inv_loss_lvl1': inv_loss_lvl1,
            'inv_loss_lvl2': inv_loss_lvl2,
            'vicreg_loss_lvl1': vicreg_loss_lvl1,
            'vicreg_loss_lvl2': vicreg_loss_lvl2,
            'var_pred_lvl1': embedding_variance_lvl1,
            'var_pred_lvl2': embedding_variance_lvl2,
            'patient_representation': patient_representation,
            'logit_context': prediction_lvl2,
            'task_loss': task_loss,
            'token_pred_loss': token_pred_loss,
            'sae_loss': sae_loss,
            'gating_weight_mean': gating_weight_mean,
            'gating_sae_fraction': gating_sae_fraction,
            'cpt_logits': cpt_logits if self.use_token_prediction_head else None,
            'icd_logits': icd_logits if self.use_token_prediction_head else None,
            'ttnc_logits': ttnc_logits if self.use_token_prediction_head else None,
        }

    def autoregressive_generation(self, cpt_tensor, icd_tensor, ttnc_tensor):
        batch_size = cpt_tensor.size(0)

        if self.use_diffusion:
            # Condition sampling on the predicted next-claim representation
            context_lvl2 = self.context_encoder_lvl2(cpt_tensor, icd_tensor, ttnc_tensor)
            _, logit_context = self.prediction_block_lvl2(context_lvl2, ttnc_tensor)
            seq_len = self.config.max_cpt_tokens + self.config.max_icd_tokens + 1
            tokens = self.diffusion_model.generate_claim(logit_context, seq_len)
            cpt_tokens = tokens[:, : self.config.max_cpt_tokens]
            icd_tokens = tokens[
                :, self.config.max_cpt_tokens : self.config.max_cpt_tokens + self.config.max_icd_tokens
            ]
            ttnc_token = tokens[:, -1]
            return {
                'predicted_cpt_codes': cpt_tokens,
                'predicted_icd_codes': icd_tokens,
                'predicted_ttnc_code': ttnc_token,
            }

        # Obtain initial patient representation
        context_lvl2 = self.context_encoder_lvl2(cpt_tensor, icd_tensor, ttnc_tensor)
        _, logit_context = self.prediction_block_lvl2(context_lvl2, ttnc_tensor)

        # === First Pass ===
        # Sample initial CPT code using multinomial sampling
        cpt_logits_initial, _, ttnc_logits = self.logits_generator(logit_context)
        cpt_probs_initial = torch.softmax(cpt_logits_initial, dim=-1)
        initial_cpt = torch.multinomial(cpt_probs_initial, num_samples=1).squeeze(1)  # [batch_size]

        # Update context with initial CPT embedding
        cpt_embeds = self.target_encoder_lvl2.cpt_embedding(initial_cpt)
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

        diffusion_loss = 0
        if self.use_diffusion and hasattr(self, 'diffusion_model'):
            cpt_tokens = batch[0][:, -1, :]
            icd_tokens = batch[1][:, -1, :]
            ttnc_tokens = batch[2][:, -1]
            tokens = torch.cat([cpt_tokens, icd_tokens, ttnc_tokens.unsqueeze(1)], dim=1)
            diffusion_loss = self.diffusion_model.forward(
                tokens,
                condition=outputs['logit_context'],
            )

        # Total loss calculation
        total_loss, task_loss = self.calculate_total_loss(
            vicreg_loss_lvl1=outputs['vicreg_loss_lvl1'],
            vicreg_loss_lvl2=outputs['vicreg_loss_lvl2'],
            task_loss=outputs['task_loss'],
            lvl2_weight=self.level_2_weight,
            token_pred_loss=outputs['token_pred_loss'],
            sae_loss=outputs['sae_loss'],
            diffusion_loss=diffusion_loss,
        )
        
        # Precision-weighted VICReg-L2 for logging
        clamped = torch.clamp(self.log_vars['vicreg_lvl2'], min=-3, max=1)
        precision_vicreg_lvl2 = torch.exp(-clamped)
        weighted_vicreg_lvl2 = (
            outputs['vicreg_loss_lvl2'] * precision_vicreg_lvl2 * self.level_2_weight
        )
        if self.level_2_weight > 0:
            self.log(
                "vicreg_lvl2",
                weighted_vicreg_lvl2,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        # --- Logging ---
        self.log('loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.use_diffusion and self.diffusion_weight > 0:
            self.log('diffusion_loss', diffusion_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Preserve existing logging
        if self.use_token_prediction_head:
            self.log('token_pred_loss', outputs['token_pred_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.use_sparse_autoencoder and self.sae_weight > 0:
            self.log('sae_loss', outputs['sae_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # Track SAE loss for epoch-level logging
            self.sae_loss_total += outputs['sae_loss'].item()
            self.sae_loss_count += 1
        
        if self.use_level1:
            self.log('Iloss1', outputs['inv_loss_lvl1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('Var1', outputs['var_pred_lvl1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.level_2_weight > 0:
            self.log('Iloss2', outputs['inv_loss_lvl2'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('Var2', outputs['var_pred_lvl2'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.use_predictor_head:
            self.log('task_loss', outputs['task_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Dynamically print active losses for quick inspection
        active_losses = [("total", total_loss)]
        if self.level_2_weight > 0:
            active_losses.append(("vicreg_lvl2", weighted_vicreg_lvl2))
        if self.use_sparse_autoencoder and self.sae_weight > 0:
            active_losses.append(("sae", outputs['sae_loss']))
        if self.use_token_prediction_head:
            active_losses.append(("token_pred", outputs['token_pred_loss']))
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

    def on_train_epoch_start(self):
        """Freeze or unfreeze log variance parameters based on epoch."""
        frozen = self.current_epoch < self.warmup_logvar_epochs
        for p in self.log_vars.parameters():
            p.requires_grad = not frozen
        self.log("logvar_frozen", float(frozen), prog_bar=True, logger=True)

    def on_after_backward(self):
        if not self._grad_check_done:
            for name, param in self.named_parameters():
                if "encoder_lvl" in name:
                    _ = param.grad
            self._grad_check_done = True


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

        if self.vicreg_batch_count > 0 and self.level_2_weight > 0:
            avg_raw = self.vicreg_lvl2_raw_total / self.vicreg_batch_count
            avg_wgt = self.vicreg_lvl2_wgt_total / self.vicreg_batch_count
            self.log('vicreg_lvl2_raw', avg_raw, prog_bar=True, logger=True)
            self.log('vicreg_lvl2_wgt', avg_wgt, prog_bar=True, logger=True)
            self.vicreg_lvl2_raw_total = 0.0
            self.vicreg_lvl2_wgt_total = 0.0
            self.vicreg_batch_count = 0

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
                self.log("val_rmse", avg_rmse, on_epoch=True, prog_bar=True, logger=True)

        active_logvars = {}
        if self.use_level1:
            active_logvars['vicreg_lvl1'] = self.log_vars['vicreg_lvl1']
        if self.level_2_weight > 0:
            active_logvars['vicreg_lvl2'] = self.log_vars['vicreg_lvl2']
        if self.use_predictor_head:
            active_logvars['task'] = self.log_vars['task']
        if self.use_token_prediction_head:
            active_logvars['token_pred'] = self.log_vars['token_pred']
        if self.use_sparse_autoencoder and self.sae_weight > 0:
            active_logvars['sae'] = self.log_vars['sae']
        if self.use_diffusion and self.diffusion_weight > 0:
            active_logvars['diffusion'] = self.log_vars['diffusion']

        for name, param in active_logvars.items():
            clamped = torch.clamp(param, min=-3, max=1)
            precision = torch.exp(-clamped).item()
            self.log(f"logvar_{name}", param.item(), prog_bar=True, logger=True)
            self.log(f"prec_{name}", precision, prog_bar=True, logger=True)

            if clamped.item() in (-3.0, 1.0):
                self.edge_streaks[name] += 1
                if self.edge_streaks[name] >= 3:
                    warnings.warn(
                        f"precision for {name} stuck at clamp edge for {self.edge_streaks[name]} epochs"
                    )
            else:
                self.edge_streaks[name] = 0

        if self.use_diffusion and self.diffusion_weight > 0:
            clamped = torch.clamp(self.log_vars['diffusion'], min=-3, max=1)
            lr_mult = torch.exp(-clamped)
            self.log("diff_lr_mult", lr_mult.item(), prog_bar=True, logger=True)

        self.repr_accumulator.clear()
        self.target_accumulator.clear()

    def configure_optimizers(self):
        # Parameters divided into encoder adapters and generator modules
        adapter_params = []
        generator_params = []
        logvar_params = []

        def collect_params(module, into_list):
            for p in module.parameters():
                if p.requires_grad:
                    into_list.append(p)

        if self.use_level1:
            collect_params(self.context_encoder_lvl1, adapter_params)
            collect_params(self.target_encoder_lvl1, adapter_params)
            collect_params(self.prediction_block_lvl1, generator_params)

        collect_params(self.context_encoder_lvl2, adapter_params)
        collect_params(self.target_encoder_lvl2, adapter_params)
        collect_params(self.prediction_block_lvl2, generator_params)

        if self.use_sparse_autoencoder:
            collect_params(self.sparse_autoencoder, generator_params)
            if self.use_gated_fusion:
                collect_params(self.sae_to_embed, generator_params)
                collect_params(self.gating_network, generator_params)

        collect_params(self.log_vars, logvar_params)
        if self.use_token_prediction_head:
            collect_params(self.logits_generator, generator_params)
            generator_params.append(self.threshold)
            generator_params.append(self.lambda_entropy)

        if self.use_predictor_head:
            collect_params(self.non_linear_predictor, generator_params)

        diffusion_params = []
        if self.use_diffusion and hasattr(self, "diffusion_model"):
            collect_params(self.diffusion_model, diffusion_params)

        param_groups = []
        if adapter_params:
            param_groups.append({
                'params': adapter_params,
                'lr': self.adapter_lr,
                'weight_decay': 1e-4,
            })
        if generator_params:
            param_groups.append({
                'params': generator_params,
                'lr': self.generator_lr,
                'weight_decay': 1e-4,
            })
        if logvar_params:
            param_groups.append({
                'params': logvar_params,
                'lr': self.generator_lr,
                'weight_decay': 1e-3,
            })
        if diffusion_params:
            param_groups.append({
                'params': diffusion_params,
                'lr': self.generator_lr * 10,
                'weight_decay': 1e-4,
            })

        optimizer_gen = torch.optim.AdamW(param_groups)

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer_gen, step_size=2, gamma=0.5
        )

        return [optimizer_gen], [scheduler]


