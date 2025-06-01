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
    def __init__(self, embed_dim, output_dim, cpt_vocab_size, icd_vocab_size,
                 ttnc_vocab_size, max_seq_length, padding_idx=0,
                 num_layers=4, num_heads=4, ff_hidden_dim=1024, dropout=0.2,
                 rnn_type='transformer',
                 use_context_pooled_patient_representation: bool = False):
        super(Level2PredictionBlock, self).__init__()

        self.padding_idx = padding_idx
        # Older checkpoints may lack vocabulary sizes which can result in
        # zero-sized embeddings and linear layers. Guard against that by
        # ensuring each vocabulary has at least one entry.
        cpt_vocab_size = max(1, cpt_vocab_size)
        icd_vocab_size = max(1, icd_vocab_size)
        ttnc_vocab_size = max(1, ttnc_vocab_size)
        # Positional Embedding Layer
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.rnn_type = rnn_type
        self.use_context_pooled_patient_representation = use_context_pooled_patient_representation

        # TTNC Embedding Layer
        self.ttnc_embedding = nn.Embedding(ttnc_vocab_size, embed_dim, padding_idx=padding_idx)

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

        # Dropout layers for regularization
        self.dropout = nn.Dropout(dropout)

        # Using residual block
        self.fc = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.LeakyReLU(),
            ResidualBlock(embed_dim * 2),
            nn.LeakyReLU(),
            nn.Linear(embed_dim * 2, output_dim)
        )

        self.activation = nn.ReLU()

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.context_emb_norm = nn.LayerNorm(embed_dim)
        self.ttnc_emb_norm = nn.LayerNorm(embed_dim)

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

    def forward(self, context_embeddings, ttnc_tokens):
        batch_size, seq_length, _ = context_embeddings.size()

        # --- There's potentially some more interesting ways to incorporate the 
        #     time between claims and positional encoding although the latter
        #     is less important with LSTM/GRU - which have tended to fare better
        #     with these shorter sequences (< 100 claims) ---
        
        # Generate position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=context_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # [batch_size, seq_length]

        position_embeds = self.position_embedding(position_ids)  # [batch_size, seq_length, embed_dim]
        self.record_statistics('position_embeds', position_embeds)

        ttnc_embeds = self.ttnc_embedding(ttnc_tokens)  # [batch_size, seq_length, embed_dim]
        self.record_statistics('ttnc_embeds', ttnc_embeds)

        combined_positional_embeds = position_embeds + ttnc_embeds  # [batch_size, seq_length, embed_dim]

        # Combine position infused time tokens with context embeddings
        combined_positional_embeds = self.ttnc_emb_norm(combined_positional_embeds)
        context_embeddings = self.context_emb_norm(context_embeddings)
        combined_sequence = context_embeddings + combined_positional_embeds
        self.record_statistics('combined_positional_embeds', combined_positional_embeds)

        # Generate attention mask based on padding
        # Transformers and RNNs treat padding opposite of each other
        attention_mask = (ttnc_tokens == self.padding_idx)  # True where TTNC tokens are padding

        valid_token_mask = (ttnc_tokens != self.padding_idx)

        # Calculate the sequence lengths by summing valid tokens across the sequence
        sequence_lengths = valid_token_mask.sum(dim=1).cpu().to(torch.int64)

        if self.rnn_type == 'transformer':
            sequence_out = self.sequence_encoder(combined_sequence, src_key_padding_mask=attention_mask)
        elif self.rnn_type in ['lstm', 'gru']:
            packed_input = nn.utils.rnn.pack_padded_sequence(combined_sequence, sequence_lengths, batch_first=True, enforce_sorted=False)
            packed_out, _ = self.sequence_encoder(packed_input)
            sequence_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=seq_length)

        self.record_statistics('sequence_out', sequence_out)
        # Apply dropout
        #sequence_out = self.dropout(sequence_out)

        # Max and mean pooling across the output sequences
        # Consider tests for masking
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(combined_sequence)
        sequence_out_masked = sequence_out.masked_fill(attention_mask_expanded.bool(), float('-inf'))
        context_max_pool = torch.max(sequence_out_masked, dim=1).values
        sequence_out_masked_for_mean = sequence_out.masked_fill(attention_mask_expanded.bool(), 0.0)
        valid_counts = (~attention_mask).sum(dim=1, keepdim=True).clamp(min=1)
        context_mean_pool = sequence_out_masked_for_mean.sum(dim=1) / valid_counts

        context_pooled = torch.cat([context_max_pool, context_mean_pool], dim=-1)

        context_pooled = self.dropout(context_pooled)

        # Final predictions
        context_output = self.fc(context_pooled)

        # Select the patient representation based on configuration
        if self.use_context_pooled_patient_representation:
            patient_representation = self.dropout(context_pooled)
        else:
            patient_representation = self.dropout(context_mean_pool)

        return patient_representation, context_output
    

class LogitsGenerator(nn.Module):
    def __init__(self, config):
        super(LogitsGenerator, self).__init__()
        self.fc1 = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)

        # Separate output layers for each token type
        self.fc_cpt = nn.Linear(config.hidden_dim, max(1, config.cpt_vocab_size))
        self.fc_icd = nn.Linear(config.hidden_dim, max(1, config.icd_vocab_size))
        self.fc_ttnc = nn.Linear(config.hidden_dim, max(1, config.ttnc_vocab_size))
    
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

