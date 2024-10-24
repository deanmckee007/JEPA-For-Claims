# models/encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tensor_utils import masked_mean, masked_variance

class Level1Encoder(nn.Module):
    """
    Encodes CPT or ICD tokens into embeddings, applies masking, and aggregates the embeddings (mean, max, min).
    This creates a representation of an individual component within a claim - 
    so a representation of the CPTs OR ICDs on a claim.
    Args:
        cpt_vocab_size (int): Size of the CPT vocabulary.
        icd_vocab_size (int): Size of the ICD vocabulary.
        embedding_dim (int): Dimensionality of the embeddings.
        padding_idx (int): The padding index in the input tokens (default is 0).

    Forward pass:
        tokens (Tensor): Input token IDs, shape [batch_size, num_claims, num_tokens].
        token_type (str): Token type, either 'cpt' or 'icd'.

    Returns:
        Tuple[Tensor, Tensor]: Aggregated embeddings (mean, max, min) for each claim, shape [batch_size, num_claims, embedding_dim * 3].
                               Per-claim mask, shape [batch_size, num_claims], indicating which claims are valid (non-padding).
    """
    def __init__(self, cpt_vocab_size, icd_vocab_size, embedding_dim, padding_idx=0):
        super(Level1Encoder, self).__init__()
        self.padding_idx = padding_idx
        self.cpt_embedding = nn.Embedding(cpt_vocab_size, embedding_dim, padding_idx=padding_idx)
        self.icd_embedding = nn.Embedding(icd_vocab_size, embedding_dim, padding_idx=padding_idx)

    def forward(self, tokens, token_type):
        if token_type == 'cpt':
            embeds = self.cpt_embedding(tokens)
        else:
            embeds = self.icd_embedding(tokens)
        
        # Mask padding tokens
        padding_mask = tokens != self.padding_idx
        
        # Aggregation ignoring padding tokens
        sum_embeds = torch.sum(embeds * padding_mask.unsqueeze(-1), dim=2)
        valid_lengths = padding_mask.sum(dim=2).clamp(min=1)
        mean_embeds = sum_embeds / valid_lengths.unsqueeze(-1)

       # Check for sequences with all padding tokens
        all_padding_mask = ~padding_mask.any(dim=2)
        # Replace invalid positions with zeros
        max_embeds = torch.where(
            all_padding_mask.unsqueeze(-1),
            torch.zeros_like(embeds[:, :, 0, :]),
            torch.max(embeds.masked_fill(~padding_mask.unsqueeze(-1), float('-inf')), dim=2).values
        )
        min_embeds = torch.where(
            all_padding_mask.unsqueeze(-1),
            torch.zeros_like(embeds[:, :, 0, :]),
            torch.min(embeds.masked_fill(~padding_mask.unsqueeze(-1), float('inf')), dim=2).values
        )
        
        aggregated = torch.cat([mean_embeds, max_embeds, min_embeds], dim=2) 
        # Compute claim-level mask
        per_claim_mask = padding_mask.any(dim=2)  # Shape: [batch_size, num_claims]
 
        
        return aggregated, per_claim_mask


class Level2Encoder(nn.Module):
    """
    Encodes CPT, ICD, and TTNC tokens with attention pooling, token rarity, and component-level attention.
    This creates a representation of an entire claim - using all of the components.
    Args:
        cpt_vocab_size (int): Size of the CPT vocabulary.
        icd_vocab_size (int): Size of the ICD vocabulary.
        ttnc_vocab_size (int): Size of the TTNC vocabulary.
        embedding_dim (int): Dimensionality of the embeddings.
        padding_idx (int): Padding index in the token sequences.
        cpt_rarity_scores (Tensor, optional): Rarity scores for CPT tokens.
        icd_rarity_scores (Tensor, optional): Rarity scores for ICD tokens.
        ttnc_rarity_scores (Tensor, optional): Rarity scores for TTNC tokens.
        use_token_rarity (bool): Whether to use token rarity during attention pooling.
        use_code_attention (bool): Whether to use attention across tokens for each claim.
        use_variance_embeddings (bool): Whether to use variance embeddings.
        use_aggregate_attention (bool): Whether to use attention across aggregate embeddings (mean, attention, variance).
        use_component_attention (bool): Whether to use attention across CPT, ICD, and TTNC components.
        dropout (float): Dropout rate.

    Forward pass:
        cpt_tokens (Tensor): CPT token IDs, shape [batch_size, num_claims, num_cpt_tokens].
        icd_tokens (Tensor): ICD token IDs, shape [batch_size, num_claims, num_icd_tokens].
        ttnc_tokens (Tensor): TTNC token IDs, shape [batch_size, num_claims, num_ttnc_tokens].

    Returns:
        Tensor: Aggregated embeddings for each claim, shape [batch_size, num_claims, embedding_dim].
    """
    def __init__(self, cpt_vocab_size, icd_vocab_size, ttnc_vocab_size, embedding_dim, padding_idx=0, 
                 cpt_rarity_scores=None, icd_rarity_scores=None, ttnc_rarity_scores=None, 
                 use_token_rarity=True, use_code_attention=True,
                 use_variance_embeddings=True, use_aggregate_attention=True,
                 use_component_attention=True, dropout=.05):
        super(Level2Encoder, self).__init__()
        print("Initializing Level2Encoder")
        self.layer_norm = nn.LayerNorm(embedding_dim)

        self.use_token_rarity = use_token_rarity
        self.use_code_attention = use_code_attention
        self.use_variance_embeddings = use_variance_embeddings
        self.use_aggregate_attention = use_aggregate_attention
        self.use_component_attention = use_component_attention

        self.cpt_attention_norm = nn.LayerNorm(embedding_dim)
        self.icd_attention_norm = nn.LayerNorm(embedding_dim)
        self.aggregate_attention_norm = nn.LayerNorm(embedding_dim)
        self.component_attention_norm = nn.LayerNorm(embedding_dim)
        self.rescale_factor = nn.Parameter(torch.ones(1) * 0.5)
        
        self.cpt_rarity_scores = cpt_rarity_scores if cpt_rarity_scores is not None else {}
        self.icd_rarity_scores = icd_rarity_scores if icd_rarity_scores is not None else {}
        self.ttnc_rarity_scores = ttnc_rarity_scores if ttnc_rarity_scores is not None else {}

        self.cpt_embedding = nn.Embedding(cpt_vocab_size, embedding_dim, padding_idx=padding_idx)
        self.icd_embedding = nn.Embedding(icd_vocab_size, embedding_dim, padding_idx=padding_idx)
        self.ttnc_embedding = nn.Embedding(ttnc_vocab_size, embedding_dim, padding_idx=padding_idx)
        
        if self.use_code_attention:
            self.attention_weights = nn.Sequential(
                nn.Linear(embedding_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1, bias=False)
            )


        # Intra-component attention layer (attention across CPT, ICD, and TTNC)
        if self.use_component_attention :
            self.component_attention = nn.Sequential(
                nn.Linear(embedding_dim, 128),  
                nn.ReLU(),
                nn.Linear(128, 1, bias=False)
            )
        # self.multihead_attention = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=1)
        self.cpt_weight = nn.Parameter(torch.tensor(1.0))  # Scalar for CPT
        self.icd_weight = nn.Parameter(torch.tensor(1.0))  # Scalar for ICD
        self.ttnc_weight = nn.Parameter(torch.tensor(1.0))  # Scalar for TTNC

        self.dropout = nn.Dropout(dropout)

        if self.use_aggregate_attention:
            self.agg_attention_weights = nn.Sequential(
                nn.Linear(embedding_dim, 128),  
                nn.ReLU(),
                nn.Linear(128, 1, bias=False)  
            )


    def component_attention_pooling(self, cpt_agg, icd_agg, ttnc_embeds, valid_mask):
        """
        Applies attention pooling across CPT, ICD, and TTNC embeddings at the claim level.
        
        Args:
            cpt_agg (Tensor): Aggregated CPT embeddings, shape [batch_size, num_claims, emb_size].
            icd_agg (Tensor): Aggregated ICD embeddings, shape [batch_size, num_claims, emb_size].
            ttnc_embeds (Tensor): TTNC embeddings, shape [batch_size, num_claims, emb_size].
            valid_mask (Tensor): Mask indicating valid claims, shape [batch_size, num_claims].

        Returns:
            Tensor: Aggregated embeddings after applying component-level attention, shape [batch_size, num_claims, emb_size].
        """
        components = torch.stack([cpt_agg, icd_agg, ttnc_embeds], dim=2)  # [batch_size, num_claims, 3, emb_size]

        # Component-level mask (assuming all components are valid if the claim is valid)
        component_mask = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [batch_size, num_claims, 1, 1]
        component_mask = component_mask.expand(-1, -1, 3, -1)    # [batch_size, num_claims, 3, 1]

        if self.use_component_attention:
            attention_scores = self.component_attention(components)  # [batch_size, num_claims, 3, 1]
            attention_scores = attention_scores.squeeze(-1)          # [batch_size, num_claims, 3]

            # Apply mask to attention scores
            attention_scores = attention_scores.masked_fill(~valid_mask.unsqueeze(-1), -1e9)

            attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_claims, 3]

            # Apply attention weights
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, num_claims, 3, 1]
            weighted_components = components * attention_weights  # [batch_size, num_claims, 3, emb_size]
            component_attention_output = weighted_components.sum(dim=2)  # [batch_size, num_claims, emb_size]
        else:
            # Sum over components if we're not using attention
            component_attention_output = components.sum(dim=2)  # [batch_size, num_claims, emb_size]

        return component_attention_output


    def attention_pooling_on_aggregates(self, mean_embeds, attention_embeds, variance_embeds, valid_mask):
        """
        Applies attention pooling across different aggregate embeddings (mean, attention, variance).
        
        Args:
            mean_embeds (Tensor): Mean embeddings, shape [batch_size, num_claims, emb_size].
            attention_embeds (Tensor): Attention embeddings, shape [batch_size, num_claims, emb_size].
            variance_embeds (Tensor, optional): Variance embeddings, shape [batch_size, num_claims, emb_size].
            valid_mask (Tensor): Mask indicating valid claims, shape [batch_size, num_claims].

        Returns:
            Tensor: Aggregated embeddings after applying attention pooling, shape [batch_size, num_claims, emb_size].
        """
        components = [mean_embeds, attention_embeds]
        if self.use_variance_embeddings:
            components.append(variance_embeds)

        # Stack components: [batch_size, num_claims, num_components, emb_size]
        stacked_components = torch.stack(components, dim=2)

        # Create a mask for valid claims
        component_mask = valid_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, len(components), 1)  # [batch_size, num_claims, num_components, 1]

        if self.use_aggregate_attention:
            attention_scores = self.agg_attention_weights(stacked_components)  # [batch_size, num_claims, num_components, 1]
            attention_scores = attention_scores.squeeze(-1)  # [batch_size, num_claims, num_components]

            # Apply mask to attention scores: set scores of invalid claims to a large negative value
            attention_scores = attention_scores.masked_fill(~valid_mask.unsqueeze(-1), -1e9)

            attention_weights = torch.softmax(attention_scores, dim=-1)  # [batch_size, num_claims, num_components]
            # Apply attention weights
            attention_weights = attention_weights.unsqueeze(-1)  # [batch_size, num_claims, num_components, 1]
            weighted_components = stacked_components * attention_weights  # [batch_size, num_claims, num_components, emb_size]
            agg_attention_output = weighted_components.sum(dim=2)  # [batch_size, num_claims, emb_size]
        else:
            # If not using attention, sum over components directly
            agg_attention_output = stacked_components.sum(dim=2)  # [batch_size, num_claims, emb_size]

        return agg_attention_output


    def code_attention_pooling(self, embeds, padding_mask, tokens, token_rarity_scores):
        """
        Applies attention pooling across codes in a claim, optionally using token rarity scores.
        
        Args:
            embeds (Tensor): Token embeddings, shape [batch_size, num_claims, num_codes, emb_size].
            padding_mask (Tensor): Mask indicating valid tokens, shape [batch_size, num_claims, num_codes].
            tokens (Tensor): Token IDs, shape [batch_size, num_claims, num_codes].
            token_rarity_scores (Tensor): Rarity scores for tokens, optional.

        Returns:
            Tensor: Aggregated embeddings after applying attention pooling, shape [batch_size, num_claims, emb_size].
        """
        batch_size, num_claims, num_codes, emb_size = embeds.size()
        embeds = embeds.reshape(-1, num_codes, emb_size)           # [batch_size*num_claims, num_codes, emb_size]
        padding_mask = padding_mask.view(-1, num_codes)         # [batch_size*num_claims, num_codes]
        tokens = tokens.reshape(-1, num_codes)                     # [batch_size*num_claims, num_codes]

        # Convert padding_mask to float mask for calculations (1 for valid, 0 for padding)
        float_mask = padding_mask.float()

        if self.use_code_attention:
            attention_scores = self.attention_weights(embeds).squeeze(-1)  # [batch_size*num_claims, num_codes]

            if self.use_token_rarity and token_rarity_scores is not None:
                token_rarity_scores_batch = token_rarity_scores[tokens].to(embeds.device)
                # Avoid log(0) by clamping
                token_rarity_scores_batch = torch.clamp(token_rarity_scores_batch, min=1e-8)
                attention_scores = attention_scores + torch.log(token_rarity_scores_batch)

            # Apply mask by setting scores of padding tokens to a very negative value
            attention_scores = attention_scores.masked_fill(~padding_mask, -1e9)

            # Compute attention weights
            attention_weights = torch.softmax(attention_scores, dim=-1)

            # Zero out attention weights of padding tokens to ensure they have no contribution
            # This is necessary here but not for the other attention functions above because
            # we're at the lowest level and still have padding within claims
            attention_weights = attention_weights * float_mask
        else:
            # Uniform attention weights over non-masked tokens
            attention_weights = float_mask
            # Normalize attention weights so they sum to 1 over non-masked tokens
            attention_weights_sum = attention_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
            attention_weights = attention_weights / attention_weights_sum

        # Apply attention weights to embeds
        attention_weights = attention_weights.unsqueeze(-1)  # [batch_size*num_claims, num_codes, 1]
        weighted_embeds = embeds * attention_weights         # [batch_size*num_claims, num_codes, emb_size]

        # Sum over the code dimension (num_codes)
        attention_output = weighted_embeds.sum(dim=1)        # [batch_size*num_claims, emb_size]

        # Reshape back to [batch_size, num_claims, emb_size]
        attention_output = attention_output.view(batch_size, num_claims, emb_size)
        return attention_output


    def forward(self, cpt_tokens, icd_tokens, ttnc_tokens):
        """
        Forward pass for the Level 2 encoder.

        Args:
            cpt_tokens (Tensor): CPT token IDs, shape [batch_size, num_claims, num_cpt_tokens].
            icd_tokens (Tensor): ICD token IDs, shape [batch_size, num_claims, num_icd_tokens].
            ttnc_tokens (Tensor): TTNC token IDs, shape [batch_size, num_claims].

        Returns:
            Tensor: Aggregated claim-level embeddings.  We're primarily applying three attention 
            weighted aggregations.  Across codes, across statistical representations of the codes
            (mean, attention mean, variance), and then across the component representations.
        """
        device = cpt_tokens.device

        cpt_tokens = cpt_tokens.long()
        icd_tokens = icd_tokens.long()
        ttnc_tokens = ttnc_tokens.long()

        cpt_embeds = self.cpt_embedding(cpt_tokens)
        icd_embeds = self.icd_embedding(icd_tokens)
        ttnc_embeds = self.ttnc_embedding(ttnc_tokens)

        cpt_padding_mask = cpt_tokens != self.cpt_embedding.padding_idx
        icd_padding_mask = icd_tokens != self.icd_embedding.padding_idx
        ttnc_padding_mask = ttnc_tokens != self.ttnc_embedding.padding_idx

        if self.use_token_rarity and self.cpt_rarity_scores is not None:
            self.cpt_rarity_scores = self.cpt_rarity_scores.to(device)
        if self.use_token_rarity and self.icd_rarity_scores is not None:
            self.icd_rarity_scores = self.icd_rarity_scores.to(device)

        cpt_attention_embeds = self.code_attention_pooling(cpt_embeds, cpt_padding_mask, cpt_tokens, self.cpt_rarity_scores)
        icd_attention_embeds = self.code_attention_pooling(icd_embeds, icd_padding_mask, icd_tokens, self.icd_rarity_scores)

        # Masked mean and variance embeddings
        cpt_mean_embeds = masked_mean(cpt_embeds, cpt_padding_mask, dim=2)
        icd_mean_embeds = masked_mean(icd_embeds, icd_padding_mask, dim=2)

        if self.use_variance_embeddings:
            cpt_variance_embeds = masked_variance(cpt_embeds, cpt_padding_mask, dim=2)
            icd_variance_embeds = masked_variance(icd_embeds, icd_padding_mask, dim=2)
        else:
            # If not using variance embeddings, use zeros
            cpt_variance_embeds = torch.zeros_like(cpt_mean_embeds)
            icd_variance_embeds = torch.zeros_like(icd_mean_embeds)

        valid_mask = ttnc_padding_mask
        cpt_agg = self.attention_pooling_on_aggregates(cpt_mean_embeds, cpt_attention_embeds, cpt_variance_embeds, valid_mask )
        icd_agg = self.attention_pooling_on_aggregates(icd_mean_embeds, icd_attention_embeds, icd_variance_embeds, valid_mask )

        aggregated_embeddings = self.component_attention_pooling(cpt_agg, icd_agg, ttnc_embeds, ttnc_padding_mask)

        aggregated_embeddings = self.dropout(aggregated_embeddings)

        return aggregated_embeddings


