# models/prediction_blocks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Level1PredictionBlock(nn.Module):
    def __init__(self, embedding_dim):
        super(Level1PredictionBlock, self).__init__()
        # Attention weights for the components (mean, max, min)
        self.attention_weights = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1, bias=False)
        )
        self.temperature = 1.0  # Temperature for softmax

    def attention_pooling(self, embeds):
        # Compute attention scores
        attention_scores = self.attention_weights(embeds).squeeze(-1)
        # Handle potential -inf values
        attention_scores = torch.where(
            torch.isinf(attention_scores),
            torch.full_like(attention_scores, -1e9),
            attention_scores
        )
        attention_weights = torch.softmax(attention_scores / self.temperature, dim=-1)
        # Apply attention weights to the embeddings
        weighted_embeds = embeds * attention_weights.unsqueeze(-1)
        return weighted_embeds


    def forward(self, context_embeddings):
        # Split the input context_embeddings into three components (mean, max, min)
        mean_embeds, max_embeds, min_embeds = torch.split(context_embeddings, context_embeddings.size(-1) // 3, dim=-1)

        # Apply attention pooling to each component
        mean_weighted = self.attention_pooling(mean_embeds)  # Shape: [batch_size, num_claims, embedding_dim]
        max_weighted = self.attention_pooling(max_embeds)   # Shape: [batch_size, num_claims, embedding_dim]
        min_weighted = self.attention_pooling(min_embeds)  # Shape: [batch_size, num_claims, embedding_dim]

        # Concatenate the weighted components back together along the last dimension
        concatenated_output = torch.cat([mean_weighted, max_weighted, min_weighted], dim=-1)  # Shape: [batch_size, num_claims, embedding_dim * 3]

        return concatenated_output

class Level2PredictionBlock(nn.Module):
    def __init__(self, embed_dim, output_dim, ttnc_vocab_size, max_seq_length, padding_idx=0,
                 num_layers=4, num_heads=4, ff_hidden_dim=1024, dropout=0.1, rnn_type='transformer'):
        super(Level2PredictionBlock, self).__init__()

        self.padding_idx = padding_idx
        # Positional Embedding Layer
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        self.rnn_type = rnn_type

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

        # Linear layers for final prediction
        self.fc = nn.Linear(embed_dim * 2, output_dim)
        self.activation = nn.ReLU()

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, context_embeddings, ttnc_tokens, target=None):
        batch_size, seq_length, _ = context_embeddings.size()

        # Generate position indices
        position_ids = torch.arange(seq_length, dtype=torch.long, device=context_embeddings.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_length)  # [batch_size, seq_length]

        # Get positional embeddings
        position_embeds = self.position_embedding(position_ids)  # [batch_size, seq_length, embed_dim]

        # Get TTNC embeddings
        ttnc_embeds = self.ttnc_embedding(ttnc_tokens)  # [batch_size, seq_length, embed_dim]

        # Combine positional embeddings and TTNC embeddings
        combined_positional_embeds = position_embeds + ttnc_embeds  # [batch_size, seq_length, embed_dim]

        # Combine with context embeddings
        scaling_factor = context_embeddings.std() / combined_positional_embeds.std()
        scaled_positional_embeds = combined_positional_embeds * scaling_factor
        combined_sequence = self.layer_norm(context_embeddings + scaled_positional_embeds)

        # Generate attention mask based on padding
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

        # Apply dropout
        sequence_out = self.dropout(sequence_out)

        # Max and mean pooling
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand_as(combined_sequence)
        sequence_out_masked = sequence_out.masked_fill(attention_mask_expanded.bool(), float('-inf'))
        context_max_pool = torch.max(sequence_out_masked, dim=1).values
        sequence_out_masked_for_mean = sequence_out.masked_fill(attention_mask_expanded.bool(), 0.0)
        valid_counts = (~attention_mask).sum(dim=1, keepdim=True).clamp(min=1)
        context_mean_pool = sequence_out_masked_for_mean.sum(dim=1) / valid_counts

        # Concatenate pooled outputs
        context_pooled = torch.cat([context_max_pool, context_mean_pool], dim=-1)

        # Apply dropout before the fully connected layers
        context_pooled = self.dropout(context_pooled)

        # Final predictions
        context_output = self.activation(self.fc(context_pooled))

        patient_representation = context_mean_pool

        return patient_representation, context_output
