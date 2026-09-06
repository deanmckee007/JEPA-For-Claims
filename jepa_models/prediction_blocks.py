# models/prediction_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    A residual block that adds the input to the output of a fully connected layer.
    
    Args:
        dim (int): The dimensionality of the input and output.

    Forward pass:
        x (Tensor): The input tensor of shape [batch_size, dim].
        Returns a tensor of the same shape where the input is added to the output of a linear transformation.
    """
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    def forward(self, x):
        return x + self.fc(x)
            

class Level1PredictionBlock(nn.Module):
    """
    *WITHIN CLAIMS*
    A prediction block for Level 1, responsible for processing embeddings of individual claims.
    We're predicting a representation of the unseen components given the seen components.
    E.g.  Predicing a representation of the CPTs given the actual ICDs 
    Args:
        embedding_dim (int): The dimensionality of the embeddings.

    Forward pass:
        context_embeddings (Tensor): A claim component representation, shape [batch_size, num_claims, embedding_dim * 3].
        mask (Tensor, optional): A mask to zero out padded codes, shape [batch_size, num_claims].

    Returns:
        output_embeddings (Tensor): Predicted representation of the unseen claim component, shape [batch_size, num_claims, embedding_dim].
    """
    def __init__(self, embedding_dim):
        super(Level1PredictionBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim * 3 * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 3 * 2, embedding_dim * 3)
        )
    
    def forward(self, context_embeddings, mask=None):
        # context_embeddings: [batch_size, num_claims, embedding_dim * 3]
        
        # Optionally apply mask to zero out embeddings for padded claims
        if mask is not None:
            context_embeddings = context_embeddings * mask.unsqueeze(-1).float()
        
        output_embeddings = self.fc(context_embeddings)  # [batch_size, num_claims, embedding_dim * 3]
        
        return output_embeddings


class Level2PredictionBlock(nn.Module):
    """
    A prediction block for Level 2, processes a series of claims representations
    and returns a representation of the next, unseen claim.
    Config handles several hyperparameters, notably the method of handling the
    sequence - transformer, lstm or gru.

    Args:
        embed_dim (int): The dimensionality of the embeddings.
        output_dim (int): The dimensionality of the output.
        cpt_vocab_size (int): Size of the CPT code vocabulary.
        icd_vocab_size (int): Size of the ICD code vocabulary.
        ttnc_vocab_size (int): Size of the TTNC code vocabulary.
        max_seq_length (int): Maximum sequence length.
        padding_idx (int): Padding index used in the input sequences.
        num_layers (int): Number of layers in the sequence encoder.
        num_heads (int): Number of attention heads (for transformer).
        ff_hidden_dim (int): Dimensionality of the hidden layer in the feed-forward network.
        dropout (float): Dropout rate.
        rnn_type (str): The type of sequence encoder, either 'transformer', 'lstm', or 'gru'.
       
    Forward pass:
        context_embeddings (Tensor): The embeddings for each sequence, shape [batch_size, seq_length, embed_dim].
        ttnc_tokens (Tensor): The TTNC tokens for each sequence, shape [batch_size, seq_length].

    Returns:
        patient_representation (Tensor): The patient representation after pooling, shape [batch_size, embed_dim].
            It's the representation for all of the context claims at the patient level.
            Useful downstream as input for any patient level predictions.
        context_output (Tensor): The output predictions, shape [batch_size, output_dim].
    """
    def __init__(
        self,
        embed_dim,
        output_dim,
        cpt_vocab_size,
        icd_vocab_size,
        ttnc_vocab_size,
        max_seq_length,
        padding_idx=0,
        num_layers=4,
        num_heads=4,
        ff_hidden_dim=1024,
        dropout=0.2,
        rnn_type='transformer',
        use_context_pooled_patient_representation: bool = False,
        use_level2_dense_prediction: bool = False,
        observed_claim_k: int = 0,
        future_claim_k: int = 0,
        use_bifurcated_patient_state: bool = False,
        use_predictive_state_bottleneck: bool = False,
        predictive_state_bottleneck_dim: int | None = None,
        use_patient_state_mean_residual: bool = False,
        patient_state_mean_residual_weight: float = 0.5,
        use_dense_decoder_bottleneck: bool = False,
        dense_decoder_bottleneck_dim: int | None = None,
        decoder_type: str = "dense",
        use_ttnc: bool = True,
        use_ttnc_ordinal_embedding: bool = False,
        ttnc_ordinal_values=None,
    ):
        super(Level2PredictionBlock, self).__init__()

        self.padding_idx = padding_idx
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.decoder_input_dim = embed_dim * 2
        self.patient_representation_dim = (
            embed_dim * 2 if use_context_pooled_patient_representation else embed_dim
        )
        # Positional Embedding Layer
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.rnn_type = rnn_type
        self.use_context_pooled_patient_representation = use_context_pooled_patient_representation
        self.use_level2_dense_prediction = use_level2_dense_prediction
        self.observed_claim_k = observed_claim_k
        self.future_claim_k = future_claim_k
        self.use_bifurcated_patient_state = use_bifurcated_patient_state
        self.use_predictive_state_bottleneck = (
            use_bifurcated_patient_state and use_predictive_state_bottleneck
        )
        self.use_patient_state_mean_residual = use_patient_state_mean_residual
        self.patient_state_mean_residual_weight = patient_state_mean_residual_weight
        self.use_dense_decoder_bottleneck = (
            use_level2_dense_prediction and use_dense_decoder_bottleneck
        )
        self.decoder_type = decoder_type
        self.use_ttnc = use_ttnc
        self.use_ttnc_ordinal_embedding = use_ttnc_ordinal_embedding
        self.num_prediction_slots = (
            observed_claim_k + future_claim_k + 1 if use_level2_dense_prediction else 1
        )
        self.dense_decoder_bottleneck_dim = (
            dense_decoder_bottleneck_dim
            if dense_decoder_bottleneck_dim is not None
            else output_dim
        )
        self.predictive_state_dim = (
            predictive_state_bottleneck_dim
            if (
                self.use_predictive_state_bottleneck
                and predictive_state_bottleneck_dim is not None
            )
            else self.decoder_input_dim
        )

        # TTNC Embedding Layer
        self.ttnc_embedding = nn.Embedding(ttnc_vocab_size, embed_dim, padding_idx=padding_idx)
        ordinal_values = (
            torch.zeros(ttnc_vocab_size, dtype=torch.float32)
            if ttnc_ordinal_values is None
            else torch.as_tensor(ttnc_ordinal_values, dtype=torch.float32)
        )
        self.register_buffer("ttnc_ordinal_values", ordinal_values)
        self.ttnc_ordinal_projection = (
            nn.Linear(1, embed_dim, bias=False)
            if use_ttnc_ordinal_embedding
            else None
        )

        if self.rnn_type == 'transformer':
            # Transformer Encoder layers
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_hidden_dim,
                dropout=dropout,
                batch_first=True
            )
            self.sequence_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        elif self.rnn_type == 'lstm':
            self.sequence_encoder = nn.LSTM(
                input_size=embed_dim,
                hidden_size=embed_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )

        elif self.rnn_type == 'gru':
            self.sequence_encoder = nn.GRU(
                input_size=embed_dim,
                hidden_size=embed_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=False
            )
        elif self.rnn_type == 'deepsets':
            layers = []
            for _ in range(num_layers):
                layers.extend([nn.Linear(embed_dim, 3 * embed_dim), nn.GELU(),
                    nn.Linear(3 * embed_dim, embed_dim), nn.LayerNorm(embed_dim)])
            self.sequence_encoder = nn.Sequential(*layers)
            self.position_embedding.weight.requires_grad_(False)
        else:
            raise ValueError(f"Unsupported rnn_type={self.rnn_type!r}")

        # Dropout layers for regularization
        self.dropout = nn.Dropout(dropout)
        if self.use_bifurcated_patient_state:
            # Keep a dedicated predictive state so dense future supervision can
            # specialize without forcing the exposed patient embedding through
            # the same decoder path.
            self.predictive_state_proj = nn.Sequential(
                nn.LayerNorm(self.decoder_input_dim),
                nn.Linear(self.decoder_input_dim, self.decoder_input_dim),
                nn.LeakyReLU(),
                ResidualBlock(self.decoder_input_dim),
                nn.LeakyReLU(),
            )
        else:
            self.predictive_state_proj = None

        if self.use_predictive_state_bottleneck:
            self.predictive_state_bottleneck = nn.Sequential(
                nn.LayerNorm(self.decoder_input_dim),
                nn.Linear(self.decoder_input_dim, self.predictive_state_dim),
                nn.LeakyReLU(),
                ResidualBlock(self.predictive_state_dim),
                nn.LeakyReLU(),
            )
        else:
            self.predictive_state_bottleneck = None

        if self.use_patient_state_mean_residual:
            self.patient_state_mean_residual_proj = nn.Linear(
                embed_dim,
                self.patient_representation_dim,
                bias=False,
            )
            self._initialize_patient_state_mean_residual()
            self.patient_state_mean_residual_norm = nn.LayerNorm(
                self.patient_representation_dim
            )
        else:
            self.patient_state_mean_residual_proj = None
            self.patient_state_mean_residual_norm = None

        if self.decoder_type == "cross_attention":
            self.slot_queries = nn.Parameter(
                torch.randn(self.num_prediction_slots, embed_dim) * 0.02
            )
            self.slot_cross_attention = nn.MultiheadAttention(
                embed_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.slot_norm = nn.LayerNorm(embed_dim)
            self.slot_ff = nn.Sequential(
                nn.Linear(embed_dim, ff_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_hidden_dim, embed_dim),
            )
            self.slot_output = nn.Linear(embed_dim, output_dim)
        else:
            self.slot_queries = None

        # Using residual block
        if self.use_dense_decoder_bottleneck:
            self.dense_decoder_bottleneck = nn.Sequential(
                nn.LayerNorm(self.predictive_state_dim),
                nn.Linear(self.predictive_state_dim, self.dense_decoder_bottleneck_dim),
                nn.LeakyReLU(),
                ResidualBlock(self.dense_decoder_bottleneck_dim),
                nn.LeakyReLU(),
            )
            self.fc = nn.Linear(
                self.dense_decoder_bottleneck_dim,
                output_dim * self.num_prediction_slots,
            )
        else:
            self.dense_decoder_bottleneck = None
            self.fc = nn.Sequential(
                nn.LayerNorm(self.predictive_state_dim),
                nn.Linear(self.predictive_state_dim, self.predictive_state_dim),
                nn.LeakyReLU(),
                ResidualBlock(self.predictive_state_dim),
                nn.LeakyReLU(),
                nn.Linear(self.predictive_state_dim, output_dim * self.num_prediction_slots)
            )

        self.activation = nn.ReLU()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.context_emb_norm = nn.LayerNorm(embed_dim)
        self.ttnc_emb_norm = nn.LayerNorm(embed_dim)

    def _initialize_patient_state_mean_residual(self):
        with torch.no_grad():
            self.patient_state_mean_residual_proj.weight.zero_()
            if self.patient_representation_dim == self.embed_dim:
                self.patient_state_mean_residual_proj.weight.copy_(torch.eye(self.embed_dim))
                return
            eye = torch.eye(self.embed_dim)
            self.patient_state_mean_residual_proj.weight[: self.embed_dim].copy_(eye)
            self.patient_state_mean_residual_proj.weight[self.embed_dim :].copy_(eye)

    def record_statistics(self, name, tensor):
        """Records the mean and standard deviation of a tensor.
           Occasionally helpful for diagnostics.
        """
        if not hasattr(self, 'statistics'):
            self.statistics = {}

        # Record the mean and standard deviation for the given tensor
        self.statistics[name] = {
            'mean': tensor.mean().item(),
            'std': tensor.std().item()
        }

    def on_epoch_end(self):
        """Handle epoch-end operations like printing statistics."""
        if hasattr(self, 'statistics'):
            # print("Statistics at the end of the epoch in Prediction Block:")
            # for name, stats in self.statistics.items():
            #     print(f'{name}: mean={stats["mean"]}, std={stats["std"]}')
            # Clear the statistics after printing
            self.statistics.clear()

    def _run_packed_rnn(self, combined_sequence, valid_token_mask):
        """Run an RNN over valid claims while preserving the input layout.

        Dataset batches are left padded so the most recent claim remains at a
        stable index. ``pack_padded_sequence`` instead requires valid values to
        occupy the prefix of each row. Compact the valid claims into a temporary
        right-padded tensor, run the packed RNN, then scatter its outputs back to
        the original claim positions for downstream masking and dense targets.
        """
        batch_size, seq_length, _ = combined_sequence.shape
        sequence_lengths = valid_token_mask.sum(dim=1).to(torch.int64)
        safe_lengths = sequence_lengths.clamp(min=1)

        valid_ranks = valid_token_mask.to(torch.int64).cumsum(dim=1) - 1
        batch_indices = torch.arange(
            batch_size,
            device=combined_sequence.device,
        ).unsqueeze(1).expand(batch_size, seq_length)

        compact_sequence = combined_sequence.new_zeros(combined_sequence.shape)
        if valid_token_mask.any():
            compact_sequence[
                batch_indices[valid_token_mask],
                valid_ranks[valid_token_mask],
            ] = combined_sequence[valid_token_mask]

        packed_input = nn.utils.rnn.pack_padded_sequence(
            compact_sequence,
            safe_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_out, _ = self.sequence_encoder(packed_input)
        compact_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out,
            batch_first=True,
            total_length=seq_length,
        )

        sequence_out = compact_output.new_zeros(compact_output.shape)
        if valid_token_mask.any():
            sequence_out[valid_token_mask] = compact_output[
                batch_indices[valid_token_mask],
                valid_ranks[valid_token_mask],
            ]
        return sequence_out, sequence_lengths

    def forward(self, context_embeddings, ttnc_tokens, return_aux: bool = False):
        batch_size, seq_length, _ = context_embeddings.size()

        # --- There's potentially some more interesting ways to incorporate the 
        #     time between claims and positional encoding although the latter
        #     is less important with LSTM/GRU - which have tended to fare better
        #     with these shorter sequences (< 100 claims) ---
        
        # Generate position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=context_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # [batch_size, seq_length]

        position_embeds = self.position_embedding(position_ids)  # [batch_size, seq_length, embed_dim]
        if self.rnn_type == 'deepsets':
            position_embeds = torch.zeros_like(position_embeds)
        self.record_statistics('position_embeds', position_embeds)

        ttnc_embeds = (
            self.ttnc_embedding(ttnc_tokens)
            if self.use_ttnc
            else torch.zeros_like(position_embeds)
        )
        self.record_statistics('ttnc_embeds', ttnc_embeds)

        combined_positional_embeds = position_embeds + ttnc_embeds  # [batch_size, seq_length, embed_dim]
        if self.ttnc_ordinal_projection is not None:
            ordinal = self.ttnc_ordinal_values[ttnc_tokens].unsqueeze(-1)
            combined_positional_embeds = (
                combined_positional_embeds
                + self.ttnc_ordinal_projection(ordinal)
            )

        # Combine position infused time tokens with context embeddings
        combined_positional_embeds = self.ttnc_emb_norm(combined_positional_embeds)
        context_embeddings = self.context_emb_norm(context_embeddings)
        combined_sequence = context_embeddings + combined_positional_embeds
        self.record_statistics('combined_positional_embeds', combined_positional_embeds)

        # Generate attention mask based on padding
        # Transformers and RNNs treat padding opposite of each other
        attention_mask = (ttnc_tokens == self.padding_idx)  # True where TTNC tokens are padding

        valid_token_mask = (ttnc_tokens != self.padding_idx)

        if self.rnn_type == 'transformer':
            sequence_out = self.sequence_encoder(combined_sequence, src_key_padding_mask=attention_mask)
        elif self.rnn_type in ['lstm', 'gru']:
            sequence_out, _ = self._run_packed_rnn(
                combined_sequence,
                valid_token_mask,
            )
        elif self.rnn_type == 'deepsets':
            sequence_out = self.sequence_encoder(combined_sequence)

        self.record_statistics('sequence_out', sequence_out)
        # Apply dropout
        #sequence_out = self.dropout(sequence_out)

        # Max and mean pooling across the output sequences
        # Consider tests for masking
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(combined_sequence)
        sequence_out_masked = sequence_out.masked_fill(attention_mask_expanded.bool(), float('-inf'))
        context_max_pool = torch.max(sequence_out_masked, dim=1).values
        has_valid_claim = valid_token_mask.any(dim=1, keepdim=True)
        context_max_pool = torch.where(
            has_valid_claim,
            context_max_pool,
            torch.zeros_like(context_max_pool),
        )
        sequence_out_masked_for_mean = sequence_out.masked_fill(attention_mask_expanded.bool(), 0.0)
        valid_counts = (~attention_mask).sum(dim=1, keepdim=True).clamp(min=1)
        context_mean_pool = sequence_out_masked_for_mean.sum(dim=1) / valid_counts

        context_pooled = torch.cat([context_max_pool, context_mean_pool], dim=-1)

        context_pooled = self.dropout(context_pooled)

        # Final predictions
        decoder_latent = None
        predictive_state = context_pooled
        if self.predictive_state_proj is not None:
            predictive_state = self.predictive_state_proj(context_pooled)
        predictive_state_base = predictive_state
        if self.predictive_state_bottleneck is not None:
            predictive_state = self.predictive_state_bottleneck(predictive_state)

        decoder_input = predictive_state
        if self.dense_decoder_bottleneck is not None:
            # Dense observed+next decoding benefits from a narrower latent than
            # the pooled patient state. This keeps local fidelity pressure from
            # collapsing into a near-identity decoder over recent claims.
            decoder_latent = self.dense_decoder_bottleneck(predictive_state)
            decoder_input = self.dropout(decoder_latent)

        if self.decoder_type == "cross_attention":
            queries = self.slot_queries.unsqueeze(0).expand(batch_size, -1, -1)
            safe_attention_mask = attention_mask.clone()
            empty_context = safe_attention_mask.all(dim=1)
            safe_attention_mask[empty_context, 0] = False
            slot_state, _ = self.slot_cross_attention(
                queries,
                sequence_out,
                sequence_out,
                key_padding_mask=safe_attention_mask,
            )
            slot_state = self.slot_norm(slot_state + self.slot_ff(slot_state))
            context_output = self.slot_output(slot_state)
            context_output[empty_context] = 0
            if self.num_prediction_slots == 1:
                context_output = context_output[:, 0]
        else:
            context_output = self.fc(decoder_input)
            if self.num_prediction_slots > 1:
                context_output = context_output.view(
                    batch_size,
                    self.num_prediction_slots,
                    self.output_dim,
                )

        # Select the patient representation based on configuration
        if self.use_context_pooled_patient_representation:
            patient_representation = self.dropout(context_pooled)
        else:
            patient_representation = self.dropout(context_mean_pool)

        patient_mean_residual = None
        if self.patient_state_mean_residual_proj is not None:
            # Preserve a direct mean-pooled path into the exposed patient state,
            # while letting the predictive branch specialize independently.
            patient_mean_residual = self.patient_state_mean_residual_proj(context_mean_pool)
            patient_representation = self.patient_state_mean_residual_norm(
                patient_representation
                + self.patient_state_mean_residual_weight * patient_mean_residual
            )

        if not return_aux:
            return patient_representation, context_output

        aux_outputs = {
            'sequence_output': sequence_out,
            'valid_token_mask': valid_token_mask,
            'context_pooled': context_pooled,
            'context_mean_pool': context_mean_pool,
            'context_max_pool': context_max_pool,
            'predictive_state_base': predictive_state_base,
            'predictive_state': predictive_state,
            'patient_mean_residual': patient_mean_residual,
            'dense_decoder_latent': decoder_latent,
        }
        return patient_representation, context_output, aux_outputs
    

class MaskedClaimQueryHead(nn.Module):
    """Predict a hidden historical claim from the remaining sequence states."""

    def __init__(self, embed_dim, output_dim, max_seq_length, num_heads=4, dropout=0.0):
        super().__init__()
        self.base_query = nn.Parameter(torch.randn(embed_dim) * 0.02)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.output = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, output_dim),
        )

    def forward(self, sequence_output, key_padding_mask, masked_positions):
        batch_size = sequence_output.size(0)
        query = self.base_query.view(1, 1, -1).expand(batch_size, 1, -1)
        query = query + self.position_embedding(masked_positions).unsqueeze(1)
        safe_mask = key_padding_mask.clone()
        safe_mask[
            torch.arange(batch_size, device=sequence_output.device), masked_positions
        ] = True
        all_masked = safe_mask.all(dim=1)
        safe_mask[all_masked, 0] = False
        state, _ = self.cross_attention(
            query,
            sequence_output,
            sequence_output,
            key_padding_mask=safe_mask,
        )
        return self.output(self.norm(state[:, 0]))


class MultiHypothesisFutureHead(nn.Module):
    """Generate several possible next-claim latents from one patient state."""

    def __init__(self, input_dim, output_dim, num_hypotheses, dropout=0.0):
        super().__init__()
        self.num_hypotheses = num_hypotheses
        self.output_dim = output_dim
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_hypotheses * output_dim),
        )

    def forward(self, patient_state):
        return self.network(patient_state).view(
            patient_state.size(0), self.num_hypotheses, self.output_dim
        )


class LogitsGenerator(nn.Module):
    def __init__(self, config):
        super(LogitsGenerator, self).__init__()
        self.fc1 = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        
        # Separate output layers for each token type
        self.fc_cpt = nn.Linear(config.hidden_dim, config.cpt_vocab_size)
        self.fc_icd = nn.Linear(config.hidden_dim, config.icd_vocab_size)
        self.fc_ttnc = nn.Linear(config.hidden_dim, config.ttnc_vocab_size)
    
    def forward(self, combined_representation):
        """
        Forward pass for generating logits.
        
        Args:
            combined_representation (Tensor): Combined representation of generated sets. Shape: [batch_size, embedding_dim]
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Logits for CPT, ICD, and TTNC.
        """
        hidden = self.relu(self.fc1(combined_representation))
        hidden = self.dropout(hidden)
        
        cpt_logits = self.fc_cpt(hidden)  # Shape: [batch_size, cpt_vocab_size]
        icd_logits = self.fc_icd(hidden)  # Shape: [batch_size, icd_vocab_size]
        ttnc_logits = self.fc_ttnc(hidden)  # Shape: [batch_size, ttnc_vocab_size]
        
        return cpt_logits, icd_logits, ttnc_logits


class MaskedClaimTokenDecoder(nn.Module):
    """
    Decode an ordered subset of next-claim tokens from a predicted claim latent.

    The target claim tokens are assumed to be positionally stable for this
    auxiliary task (for example because they have been sorted
    alphanumerically). We supervise only masked target positions, keeping the
    latent-space JEPA objective primary.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        cpt_vocab_size,
        icd_vocab_size,
        ttnc_vocab_size,
        max_cpt_tokens,
        max_icd_tokens,
        dropout=0.0,
        include_ttnc=False,
    ):
        super(MaskedClaimTokenDecoder, self).__init__()
        self.include_ttnc = include_ttnc
        self.context_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.cpt_position_embeddings = nn.Embedding(max_cpt_tokens, hidden_dim)
        self.icd_position_embeddings = nn.Embedding(max_icd_tokens, hidden_dim)
        self.cpt_head = nn.Linear(hidden_dim, cpt_vocab_size)
        self.icd_head = nn.Linear(hidden_dim, icd_vocab_size)
        self.dropout = nn.Dropout(dropout)

        if self.include_ttnc:
            self.ttnc_head = nn.Linear(hidden_dim, ttnc_vocab_size)
        else:
            self.ttnc_head = None

    def _decode_ordered_tokens(self, base_hidden, position_embeddings, output_head):
        batch_size = base_hidden.size(0)
        num_positions = position_embeddings.num_embeddings
        position_ids = torch.arange(
            num_positions,
            device=base_hidden.device,
        ).unsqueeze(0).expand(batch_size, num_positions)
        position_hidden = position_embeddings(position_ids)
        token_hidden = self.dropout(base_hidden.unsqueeze(1) + position_hidden)
        return output_head(token_hidden)

    def forward(self, claim_latent):
        base_hidden = self.context_proj(claim_latent)
        cpt_logits = self._decode_ordered_tokens(
            base_hidden,
            self.cpt_position_embeddings,
            self.cpt_head,
        )
        icd_logits = self._decode_ordered_tokens(
            base_hidden,
            self.icd_position_embeddings,
            self.icd_head,
        )
        ttnc_logits = self.ttnc_head(base_hidden) if self.include_ttnc else None
        return cpt_logits, icd_logits, ttnc_logits

