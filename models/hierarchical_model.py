# models/hierarchical_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from models.encoders import Level1Encoder, Level2Encoder
from models.prediction_blocks import Level1PredictionBlock, Level2PredictionBlock, LogitsGenerator
from models.sparse_autoencoder import SparseAutoencoder
from utils.metrics import calculate_rmse
from utils.tensor_utils import calculate_entropy
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
    - discriminator (nn.Sequential): Discriminator used for adversarial training.
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
    
    calculate_total_loss(vicreg_loss_lvl1, vicreg_loss_lvl2, task_loss, lvl2_weight, token_pred_loss, adversarial_loss):
        Combines VICReg, task, token prediction, and adversarial losses into a total loss.
        - Parameters:
            - vicreg_loss_lvl1 (Tensor): VICReg loss at Level 1.
            - vicreg_loss_lvl2 (Tensor): VICReg loss at Level 2.
            - task_loss (Tensor): Loss for any regression tasks (optional).
            - lvl2_weight (float): Weight for the Level 2 VICReg loss.
            - token_pred_loss (Tensor): Loss for token prediction tasks (optional).
            - adversarial_loss (Tensor): Loss from adversarial training (optional).
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
        Defines the training step, including manual optimization for generator and discriminator.
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
        Configures optimizers for the generator and discriminator, with different learning rates and weight decay for specific parameters.
    """
    def __init__(self, config):
        super(HierarchicalClaimsModel, self).__init__()
        self.automatic_optimization = True
        self.save_hyperparameters()
        print("Initializing HierarchicalClaimsModel")
        # Check config values
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
        self.cpt_vocab_size=config.cpt_vocab_size,
        self.icd_vocab_size=config.icd_vocab_size,

        self.accumulated_representations = []
        self.accumulated_targets = []
        self.regression_weights = None
        self.alternate_flag = True
        # Initialize SAE loss tracking
        self.sae_loss_total = 0.0
        self.sae_loss_count = 0

        self.threshold = nn.Parameter(torch.tensor(0.1))
        self.lambda_entropy = nn.Parameter(torch.tensor(config.lambda_entropy))
        self.logits_generator = LogitsGenerator(config)

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
        )

        if self.use_sparse_autoencoder:
            self.sparse_autoencoder = SparseAutoencoder(
                input_dim=config.embedding_dim,
                hidden_dim=config.sae_hidden_dim,
                k=config.sae_k,
            )
            if self.use_gated_fusion:
                self.sae_to_embed = nn.Linear(config.sae_hidden_dim, config.embedding_dim)
                self.gating_network = nn.Sequential(
                    nn.Linear(config.embedding_dim + config.embedding_dim, config.embedding_dim),
                    nn.Sigmoid()
                )

        self.lr = config.lr
        self.loss_fn = nn.MSELoss()
        # Adjust the size of log_vars since we're removing the adversarial component
        self.log_vars = nn.ParameterDict({
            'vicreg_lvl1': nn.Parameter(torch.zeros(1)),
            'vicreg_lvl2': nn.Parameter(torch.zeros(1)),
            'task': nn.Parameter(torch.zeros(1)),
            'token_pred': nn.Parameter(torch.zeros(1)),
            'sae': nn.Parameter(torch.zeros(1)),
            #'adversarial': nn.Parameter(torch.zeros(1)),
        })

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

        # self.discriminator = nn.Sequential(
        #     nn.Linear(config.embedding_dim, config.hidden_dim),
        #     nn.LeakyReLU(0.2),
        #     nn.Dropout(config.dropout),
        #     nn.Linear(config.hidden_dim, 1),
        #     nn.Sigmoid()
        # )
        
        # # Initialize discriminator parameters
        # self._initialize_weights(self.discriminator)

        self.initialize_target_encoders()

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

    def calculate_total_loss(self, vicreg_loss_lvl1, vicreg_loss_lvl2, task_loss, lvl2_weight, token_pred_loss, sae_loss=0):
        # Conceptually - https://arxiv.org/pdf/1705.07115
        log_var_denom = 2
        log_var_denom += 1 if self.use_level1 else 0
        log_var_denom += 1 if self.use_predictor_head else 0
        log_var_denom += 1 if self.use_token_prediction_head else 0
        log_var_denom += 1 if self.use_sparse_autoencoder else 0


        clamped_log_vars = {k: torch.clamp(v, min=-10, max=10) for k, v in self.log_vars.items()}
        precision_vicreg_lvl2 = torch.exp(clamped_log_vars['vicreg_lvl2'])

        if self.use_level1:
            precision_vicreg_lvl1 = torch.exp(clamped_log_vars['vicreg_lvl1'])
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
            precision_task = torch.exp(clamped_log_vars['task'])
            weighted_task_loss = task_loss * precision_task
            total_loss = total_loss + weighted_task_loss
            total_precision = total_precision + precision_task
            total_log_var = total_log_var + clamped_log_vars['task']

        if self.use_token_prediction_head:
            precision_token = torch.exp(clamped_log_vars['token_pred'])
            weighted_token_loss = token_pred_loss * precision_token
            total_loss = total_loss + weighted_token_loss
            total_precision = total_precision + precision_token
            total_log_var = total_log_var + clamped_log_vars['token_pred']

        if self.use_sparse_autoencoder:
            precision_sae = torch.exp(clamped_log_vars['sae'])
            weighted_sae_loss = sae_loss * precision_sae
            total_loss = total_loss + weighted_sae_loss
            total_precision = total_precision + precision_sae
            total_log_var = total_log_var + clamped_log_vars['sae']

            # precision_adv = torch.exp(clamped_log_vars['adversarial'])
            # total_loss += adversarial_loss * precision_adv
            # total_precision += precision_adv
            # total_log_var += clamped_log_vars['adversarial']

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

    # def adversarial_loss(self, y_hat, y):
    #     # Binary cross-entropy loss
    #     return nn.functional.binary_cross_entropy(y_hat, y)

    # def reset_generated_sets(self, batch_size):
    #     """
    #     Initialize empty generated sets for a new batch.
    #     """
    #     self.generated_cpt = torch.zeros(batch_size, self.max_generated_tokens, dtype=torch.long, device=self.device)
    #     self.generated_icd = torch.zeros(batch_size, self.max_generated_tokens, dtype=torch.long, device=self.device)
    #     self.generated_ttnc = torch.zeros(batch_size, dtype=torch.long, device=self.device)

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
        if self.use_token_prediction_head:
            # Initialize generated sets
            batch_size = cpt_tensor.size(0)
            #self.reset_generated_sets(batch_size)

            logit_context = prediction_lvl2
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
                target_cpt_multi_hot = self.create_multi_hot_targets(target_cpt, self.cpt_vocab_size[0], padding_idx=0)
                target_icd_multi_hot = self.create_multi_hot_targets(target_icd, self.icd_vocab_size[0], padding_idx=0)

                # Compute loss using targets
                criterion_bce = nn.BCEWithLogitsLoss()
                criterion_ce = nn.CrossEntropyLoss()
                cpt_loss = criterion_bce(cpt_logits, target_cpt_multi_hot)
                icd_loss = criterion_bce(icd_logits, target_icd_multi_hot)
                ttnc_loss = criterion_ce(ttnc_logits, target_ttnc.squeeze(1))
                token_pred_loss = cpt_loss + icd_loss + ttnc_loss

                # Prepare fake claim tensors
            # fake_cpt_tensor = self.multi_hot_to_indices_tensor(generated_cpt)
            # fake_icd_tensor = self.multi_hot_to_indices_tensor(generated_icd)
            # fake_ttnc_tensor = generated_ttnc.unsqueeze(1)  # [batch_size, 1]

            # Pass generated fake claim through the target encoder
            # fake_target_lvl2 = self.target_encoder_lvl2(
            #     fake_cpt_tensor, fake_icd_tensor, fake_ttnc_tensor
            # ).squeeze(1)

            # fake_preds_for_generator = self.discriminator(fake_target_lvl2)
            # real_labels_for_generator = torch.ones_like(fake_preds_for_generator)
            # adversarial_loss = self.adversarial_loss(fake_preds_for_generator, real_labels_for_generator)


        sae_loss = 0
        if self.use_sparse_autoencoder:
            recon = self.sparse_autoencoder(patient_representation)
            sae_loss = F.mse_loss(recon, patient_representation)
            if self.use_gated_fusion:
                sae_encoded = self.sparse_autoencoder.encoder(patient_representation)
                sae_embed = self.sae_to_embed(sae_encoded)
                gate_input = torch.cat([patient_representation, sae_embed], dim=-1)
                gate = self.gating_network(gate_input)
                patient_representation = gate * sae_embed + (1 - gate) * patient_representation

        # Compute total loss
        total_loss, task_loss = self.calculate_total_loss(
            vicreg_loss_lvl1,
            vicreg_loss_lvl2,
            task_loss,
            lvl2_weight=self.level_2_weight,
            token_pred_loss=token_pred_loss,
            sae_loss=sae_loss,
            #adversarial_loss=adversarial_loss
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
            'task_loss': task_loss,
            'token_pred_loss': token_pred_loss,
            'sae_loss': sae_loss,
            'cpt_logits': cpt_logits if self.use_token_prediction_head else None,
            'icd_logits': icd_logits if self.use_token_prediction_head else None,
            'ttnc_logits': ttnc_logits if self.use_token_prediction_head else None,
            # 'real_target_lvl2': real_target_lvl2,
            # 'fake_target_lvl2': fake_target_lvl2, 
            # 'predicted_cpt_codes': generated_cpt,
            # 'predicted_icd_codes': generated_icd,
            # 'predicted_ttnc_code': generated_ttnc
        }

    def autoregressive_generation(self, cpt_tensor, icd_tensor, ttnc_tensor):
        batch_size = cpt_tensor.size(0)
        # self.reset_generated_sets(batch_size)

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

        # Calculate entropy of the CPT and ICD probabilities
        cpt_entropy = calculate_entropy(cpt_probs)
        icd_entropy = calculate_entropy(icd_probs)

       # Adjust threshold based on entropy (lower threshold where entropy is lower)
        # You can fine-tune this logic as needed
        dynamic_cpt_threshold = self.threshold - self.lambda_entropy * (1 - cpt_entropy / torch.log(torch.tensor(cpt_probs.size(-1))))
        dynamic_icd_threshold = self.threshold - self.lambda_entropy * (1 - icd_entropy / torch.log(torch.tensor(icd_probs.size(-1))))

        # Clamp thresholds to ensure they remain in a valid range
        dynamic_cpt_threshold = dynamic_cpt_threshold.clamp(min=0.01, max=0.9)
        dynamic_icd_threshold = dynamic_icd_threshold.clamp(min=0.01, max=0.9)

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

        # Total loss calculation
        total_loss, task_loss = self.calculate_total_loss(
            vicreg_loss_lvl1=outputs['vicreg_loss_lvl1'],
            vicreg_loss_lvl2=outputs['vicreg_loss_lvl2'],
            task_loss=outputs['task_loss'],
            lvl2_weight=self.level_2_weight,
            token_pred_loss=outputs['token_pred_loss'],
            sae_loss=outputs['sae_loss']
        )

        # --- Logging ---
        self.log('loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Preserve existing logging
        if self.use_token_prediction_head:
            self.log('token_pred_loss', outputs['token_pred_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        if self.use_sparse_autoencoder:
            self.log('sae_loss', outputs['sae_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # Track SAE loss for epoch-level logging
            self.sae_loss_total += outputs['sae_loss'].item()
            self.sae_loss_count += 1
        
        if self.use_level1:
            self.log('Iloss1', outputs['inv_loss_lvl1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('Var1', outputs['var_pred_lvl1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log('Iloss2', outputs['inv_loss_lvl2'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Var2', outputs['var_pred_lvl2'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if self.use_predictor_head:
            self.log('task_loss', outputs['task_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Update target encoders after each step
        self.update_target_encoders()
        
        # Accumulate representations and targets for regression evaluation
        self.accumulated_representations.append(outputs['patient_representation'].cpu().detach())
        self.accumulated_targets.append(batch[3].cpu().detach())
        
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

            if len(self.accumulated_representations) > 0:
                # Stack accumulated data
                X_accum = torch.cat(self.accumulated_representations)
                y_accum = torch.cat(self.accumulated_targets)

                if self.use_zero_target_mask:
                    # Filter out pairs where target is zero
                    non_zero_mask = y_accum != 0
                    X_accum = X_accum[non_zero_mask]
                    y_accum = y_accum[non_zero_mask]

                # Convert data to numpy arrays if necessary
                X_accum_np = X_accum.cpu().numpy()
                y_accum_np = y_accum.cpu().numpy()

                # Initialize KFold with 5 splits
                kf = KFold(n_splits=5, shuffle=True, random_state=42)
                rmse_list = []

                for fold, (train_index, val_index) in enumerate(kf.split(X_accum_np)):
                    # Split data into training and validation sets
                    X_train, X_val = X_accum_np[train_index], X_accum_np[val_index]
                    y_train, y_val = y_accum_np[train_index], y_accum_np[val_index]

                    # Before converting to tensors
                    scaler_X = StandardScaler()
                    X_train_np = scaler_X.fit_transform(X_train)
                    X_val_np = scaler_X.transform(X_val)
                    scaler_y = StandardScaler()
                    y_train_np = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
                    y_val_np = scaler_y.transform(y_val.reshape(-1, 1)).flatten()

                    # Convert back to tensors if needed
                    X_train_torch = torch.from_numpy(X_train_np).float().to(self.device)
                    y_train_torch = torch.from_numpy(y_train_np).float().to(self.device)
                    X_val_torch = torch.from_numpy(X_val_np).float().to(self.device)
                    y_val_torch = torch.from_numpy(y_val_np).float().to(self.device)

                    # Train linear regression on the training fold
                    self.train_linear_regression(X_train_torch, y_train_torch)

                    # Calculate RMSE on the validation fold
                    val_rmse = calculate_rmse(self.regression_weights, X_val_torch, y_val_torch, scaler_y)

                    rmse_list.append(val_rmse.item())

                # Calculate average RMSE across all folds
                avg_rmse = sum(rmse_list) / len(rmse_list)
                print(f" Average Validation RMSE: {avg_rmse}")
                self.log('val_rmse', avg_rmse, on_step=False, on_epoch=True, prog_bar=False, logger=True)
                for name, param in self.log_vars.items():
                    print(self.log(name, param.item(), prog_bar=True))

                # Reset the accumulated data
                self.accumulated_representations = []
                self.accumulated_targets = []

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
        if self.use_sparse_autoencoder and self.sae_loss_count > 0:
            avg_sae_loss = self.sae_loss_total / self.sae_loss_count
            self.log('avg_sae_loss', avg_sae_loss, prog_bar=True, logger=True)
            self.sae_loss_total = 0.0
            self.sae_loss_count = 0

    def configure_optimizers(self):
        # Generator parameters
        gen_params = []
        if self.use_level1:
            gen_params.extend(self.context_encoder_lvl1.parameters())
            gen_params.extend(self.prediction_block_lvl1.parameters())
        gen_params.extend(self.context_encoder_lvl2.parameters())
        gen_params.extend(self.prediction_block_lvl2.parameters())
        if self.use_sparse_autoencoder:
            gen_params.extend(self.sparse_autoencoder.parameters())
        gen_params.extend(self.log_vars.parameters())  # Add individual parameters from ParameterDict
        gen_params.extend(self.logits_generator.parameters())
        if self.use_predictor_head:
            gen_params.extend(self.non_linear_predictor.parameters())
        if self.use_token_prediction_head:
            gen_params.append(self.threshold)
            gen_params.append(self.lambda_entropy)
        
        # Collect the IDs of log_vars parameters for identity-based exclusion
        log_vars_ids = set(id(p) for p in self.log_vars.parameters())
        
        # Define parameter groups based on the criteria:
        # - params_with_weight_decay: requires_grad=True, ndim > 1, not in log_vars
        # - params_without_weight_decay: requires_grad=True, ndim == 1, in log_vars
        params_with_weight_decay = [
            param for param in gen_params 
            if param.requires_grad and param.ndim > 1 and id(param) not in log_vars_ids
        ]

        params_without_weight_decay = [
            param for param in gen_params 
            if param.requires_grad and param.ndim == 1 and id(param) in log_vars_ids
        ]

        # Define generator optimizer with two parameter groups
        optimizer_gen = torch.optim.AdamW(
            [
                {
                    'params': params_with_weight_decay,
                    'lr': self.lr,
                    'weight_decay': 1e-4
                },
                {
                    'params': params_without_weight_decay,
                    'lr': self.lr,
                    'weight_decay': 0
                }
            ]
        )

        # If you plan to use a discriminator, define its optimizer similarly
        # For now, it's commented out as per your current implementation
        # Define discriminator optimizer (if applicable)
        # optimizer_dis = torch.optim.AdamW(
        #     [
        #         {
        #             'params': [param for param in dis_params if param.requires_grad and param.ndim > 1],
        #             'lr': self.lr,
        #             'weight_decay': 1e-4
        #         },
        #         {
        #             'params': [param for param in dis_params if param.requires_grad and param.ndim == 1],
        #             'lr': self.lr,
        #             'weight_decay': 0
        #         }
        #     ]
        # )

        return optimizer_gen  # Return only the generator optimizer if discriminator is not used


