# models/hierarchical_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from models.encoders import Level1Encoder, Level2Encoder
from models.prediction_blocks import Level1PredictionBlock, Level2PredictionBlock
from utils.metrics import calculate_rmse

class HierarchicalClaimsModel(pl.LightningModule):
    def __init__(self, config):
        super(HierarchicalClaimsModel, self).__init__()
        self.save_hyperparameters()
        print("Initializing HierarchicalClaimsModel")
        # Check config values
        print(f"cpt_vocab_size: {config.cpt_vocab_size}")
        print(f"icd_vocab_size: {config.icd_vocab_size}")
        print(f"ttnc_vocab_size: {config.ttnc_vocab_size}")
        print(f"embedding_dim: {config.embedding_dim}")

        self.ema_decay = config.ema_decay
        self.epsilon = config.epsilon
        self.target_var = config.target_var
        self.amplification_power = config.amplification_power
        self.steps_per_epoch = config.steps_per_epoch
        self.var_penalty_scale = config.var_penalty_scale
        self.rnn_type = config.rnn_type
        self.use_predictor_head = config.use_predictor_head

        self.accumulated_representations = []
        self.accumulated_targets = []
        self.regression_weights = None
        self.alternate_flag = True

        # Initialize Encoders and Prediction Blocks
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
            use_component_attention=config.use_component_attention
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
            use_component_attention=config.use_component_attention
        )
        
        self.prediction_block_lvl2 = Level2PredictionBlock(
            embed_dim=config.rnn_hidden_dim,
            output_dim=config.output_dim,
            ttnc_vocab_size=config.ttnc_vocab_size,
            max_seq_length=config.max_claims_len,
            padding_idx=0,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_hidden_dim=config.ff_hidden_dim,
            dropout=config.dropout,
            rnn_type=config.rnn_type
        )
        
        self.lr = config.lr
        self.loss_fn = nn.MSELoss()
        self.log_vars = nn.Parameter(torch.zeros(3))

        if self.use_predictor_head:
            self.non_linear_predictor = nn.Sequential(
                nn.Linear(config.embedding_dim, config.hidden_dim),  # First layer
                nn.ReLU(),  # Non-linearity
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),  # Second layer
                nn.ReLU(),  # Non-linearity
                nn.Linear(config.hidden_dim // 2, 1)  # Output layer for regression
            )

        self.initialize_target_encoders()

    def initialize_target_encoders(self):
        self.apply(self._initialize_weights)

        # Copy context encoder parameters to target encoders and set requires_grad=False
        for param_q, param_k in zip(self.context_encoder_lvl2.parameters(), self.target_encoder_lvl2.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def _initialize_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier initialization for linear layers
            nn.init.xavier_uniform_(module.weight)
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

    def update_target_encoders(self):
        with torch.no_grad():
            for param_q, param_k in zip(self.context_encoder_lvl2.parameters(), self.target_encoder_lvl2.parameters()):
                param_k.data = self.ema_decay * param_k.data + (1.0 - self.ema_decay) * param_q.data

    def calculate_vicreg_loss(self, context_output, target_output, level):
        # Center the outputs
        context_output = context_output - context_output.mean(dim=0, keepdim=True)
        target_output = target_output - target_output.mean(dim=0, keepdim=True)

        # Variance term
        std_context = torch.sqrt(context_output.var(dim=0) + self.epsilon)
        std_target = torch.sqrt(target_output.var(dim=0) + self.epsilon)
        var_loss = (torch.mean(F.relu(self.target_var - std_context)) / 2 + 
                    torch.mean(F.relu(self.target_var - std_target)) / 2)**self.amplification_power

        # Invariance term (MSE)
        inv_loss = self.loss_fn(context_output, target_output)

        # Ignoring covariance loss for now
        if level == '1':
            vicreg_loss = (var_loss * self.var_penalty_scale + inv_loss) 
        else:
            vicreg_loss = (var_loss * self.var_penalty_scale + inv_loss)

        return vicreg_loss, var_loss, inv_loss

    def compute_closed_form_regression(self, X, y, sampled_indices=None, sample_fraction=0.5):
        if sampled_indices is None:
            sample_size = int(sample_fraction * X.size(0))
            sampled_indices = torch.randperm(X.size(0))[:sample_size]

        X_sampled = X[sampled_indices]
        y_sampled = y[sampled_indices]

        # Perform closed-form regression using the sampled data
        X_T = X_sampled.t()
        try:
            self.regression_weights = torch.inverse(X_T @ X_sampled) @ X_T @ y_sampled
        except RuntimeError as e:
            print("Matrix inversion failed", e)

        return sampled_indices  # Return sampled indices to reuse in the RMSE calculation

    def calculate_rmse(self, X_sample, y_sample):
        """Calculate RMSE on the sampled data using the trained linear regression model"""
        if self.regression_weights is None:
            return None

        y_pred = X_sample @ self.regression_weights
        mse = torch.mean((y_pred - y_sample) ** 2)
        rmse = torch.sqrt(mse)
        return rmse

    def calculate_total_loss(self, vicreg_loss_lvl1, vicreg_loss_lvl2, task_loss):
        clamped_log_vars = torch.clamp(self.log_vars, min=-10, max=10)
        precision_vicreg_lvl1 = torch.exp(clamped_log_vars[0])
        precision_vicreg_lvl2 = torch.exp(clamped_log_vars[1])

        # Weighted losses
        weighted_vicreg_lvl1_loss = vicreg_loss_lvl1 * precision_vicreg_lvl1
        weighted_vicreg_lvl2_loss = vicreg_loss_lvl2 * precision_vicreg_lvl2

        if self.use_predictor_head:
            precision_task = torch.exp(clamped_log_vars[2])

            weighted_task_loss = task_loss * precision_task

            total_loss = (weighted_vicreg_lvl1_loss + weighted_vicreg_lvl2_loss + weighted_task_loss) / 3.0  # Averaging for stability

            # Adding regularization to avoid collapsing precision to zero
            total_loss += (clamped_log_vars[0] + clamped_log_vars[1] + clamped_log_vars[2]) / 3.0
        else:
            total_loss = total_loss = (weighted_vicreg_lvl1_loss + weighted_vicreg_lvl2_loss) / 2.0
            total_loss += (clamped_log_vars[0] + clamped_log_vars[1]) / 2.0

        return total_loss, task_loss

    def forward(self, cpt_tensor, icd_tensor, ttnc_tensor, target=None):
        # Level 1: Within-claims prediction
        if self.alternate_flag:
            # Pass cpt_tensor to context_encoder_lvl1 and icd_tensor to target_encoder_lvl1
            context_lvl1 = self.context_encoder_lvl1(cpt_tensor, 'cpt')
            target_lvl1 = self.target_encoder_lvl1(icd_tensor, 'icd')
        else:
            # Pass icd_tensor to context_encoder_lvl1 and cpt_tensor to target_encoder_lvl1
            context_lvl1 = self.context_encoder_lvl1(icd_tensor, 'icd')
            target_lvl1 = self.target_encoder_lvl1(cpt_tensor, 'cpt')

        # Flip the alternation flag for the next forward pass
        self.alternate_flag = not self.alternate_flag
        
        prediction_lvl1 = self.prediction_block_lvl1(context_lvl1)

        batch_size, num_claims, embedding_dim = context_lvl1.shape
        # Reshape prediction_lvl1 and target_lvl1 to [batch_size * num_claims, embedding_dim]
        prediction_lvl1 = prediction_lvl1.view(batch_size * num_claims, embedding_dim)
        embedding_variance_lvl1 = torch.var(prediction_lvl1, dim=0).mean()
        target_lvl1 = target_lvl1.view(batch_size * num_claims, embedding_dim)
    
        vicreg_loss_lvl1, var_loss_lvl1, inv_loss_lvl1 = self.calculate_vicreg_loss(prediction_lvl1, target_lvl1, "1")

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
        
        patient_representation, prediction_lvl2 = self.prediction_block_lvl2(context_lvl2, context_ttnc, target)
        
        context_padding_mask = (context_ttnc != 0).float()
        target_padding_mask = (target_ttnc != 0).float()

        # A sequence is valid if it has at least one non-padding token in both context and target
        valid_sequences_mask = (context_padding_mask.sum(dim=1) > 0) & (target_padding_mask.sum(dim=1) > 0)
        
        # Compute variance across the combined batch_size and num_claims dimension
        embedding_variance_lvl2 = torch.var(prediction_lvl2[valid_sequences_mask], dim=0).mean()

        vicreg_loss_lvl2, var_loss_lvl2, inv_loss_lvl2 = self.calculate_vicreg_loss(
            prediction_lvl2[valid_sequences_mask],
            target_lvl2[valid_sequences_mask],
            "2"
        )

        task_loss = 0
        if self.use_predictor_head and target is not None:
            target_pred = self.non_linear_predictor(context_lvl2.mean(dim=1))
            target = target.view_as(target_pred)
            task_loss = self.loss_fn(target_pred, target)

        # Combine losses
        total_loss, task_loss = self.calculate_total_loss(vicreg_loss_lvl1, vicreg_loss_lvl2, task_loss)  # Add regression_loss if available

        return {
            'loss': total_loss,
            'var_loss_lvl1': var_loss_lvl1,
            'var_loss_lvl2': var_loss_lvl2,
            'inv_loss_lvl1': inv_loss_lvl1,
            'inv_loss_lvl2': inv_loss_lvl2,
            'vicreg_loss_lvl2': vicreg_loss_lvl2,
            'var_pred_lvl1': embedding_variance_lvl1,
            'var_pred_lvl2': embedding_variance_lvl2,
            'patient_representation': patient_representation,
            'task_loss': task_loss
        }

    def training_step(self, batch, batch_idx):
        cpt_tensor, icd_tensor, ttnc_tensor, target = batch
        outputs = self(cpt_tensor, icd_tensor, ttnc_tensor, target)
        self.accumulated_representations.append(outputs['patient_representation'].cpu().detach())
        self.accumulated_targets.append(target.cpu().detach())
        #self.log('train_loss', outputs['loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Iloss1', outputs['inv_loss_lvl1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Var1', outputs['var_pred_lvl1'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Iloss2', outputs['inv_loss_lvl2'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('Var2', outputs['var_pred_lvl2'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('task_loss', outputs['task_loss'], on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.update_target_encoders()

        return outputs
    
    def sample_accumulated_data(self, sample_percent=0.5):
        """Sample a percentage of accumulated representations and targets"""
        X = torch.cat(self.accumulated_representations)
        y = torch.cat(self.accumulated_targets)

        sample_size = int(sample_percent * X.size(0))

        sample_size = max(1, sample_size)

        indices = torch.randperm(X.size(0))[:sample_size]
        X_sample = X[indices]
        y_sample = y[indices]

        return X_sample, y_sample

    def train_linear_regression(self, X_sample, y_sample):
        """Train the linear model using sampled representations"""
        X_T = X_sample.t()
        try:
            self.regression_weights = torch.inverse(X_T @ X_sample) @ X_T @ y_sample
        except RuntimeError as e:
            print("Matrix inversion failed", e)

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx=None):
        if (batch_idx + 1) == self.steps_per_epoch:
            total_norm = 0
            for name, param in self.named_parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    print(f"Grad Norm for {name}: {param_norm.item()}")
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            print(f"Total Gradient Norm: {total_norm}")

            if len(self.accumulated_representations) > 0:
                # Stack accumulated data
                X_accum = torch.cat(self.accumulated_representations)
                y_accum = torch.cat(self.accumulated_targets)

                # Shuffle the data
                indices = torch.randperm(X_accum.size(0))
                X_accum = X_accum[indices]
                y_accum = y_accum[indices]

                # Split into training (50%) and validation (50%)
                # Current target is noisy - typically an 80/20 split
                split_idx = int(X_accum.size(0) * 0.5)
                X_train, X_val = X_accum[:split_idx], X_accum[split_idx:]
                y_train, y_val = y_accum[:split_idx], y_accum[split_idx:]

                self.train_linear_regression(X_train, y_train)

                val_rmse = calculate_rmse(self.regression_weights, X_val, y_val)
                print(f"Validation RMSE: {val_rmse.item() if val_rmse is not None else 'N/A'}")
                self.log('val_rmse', val_rmse, on_step=False, on_epoch=True, prog_bar=True, logger=True)

                # Reset the accumulated data
                self.accumulated_representations = []
                self.accumulated_targets = []

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([
            {'params': [param for name, param in self.named_parameters() if "bias" not in name and "layer_norm" not in name], 'lr': self.lr, 'weight_decay': 1e-4},
            {'params': [param for name, param in self.named_parameters() if "bias" in name or "layer_norm" in name], 'lr': self.lr, 'weight_decay': 0}
        ])
        return [optimizer]
