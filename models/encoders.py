# models/encoders.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tensor_utils import masked_mean, masked_variance

class Level1Encoder(nn.Module):
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
        
        return aggregated


class Level2Encoder(nn.Module):
    def __init__(self, cpt_vocab_size, icd_vocab_size, ttnc_vocab_size, embedding_dim, padding_idx=0, 
                 cpt_rarity_scores=None, icd_rarity_scores=None, ttnc_rarity_scores=None, 
                 use_token_rarity=True, use_code_attention=True,
                 use_variance_embeddings=True, use_aggregate_attention=True,
                 use_component_attention=True):
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

        if self.use_aggregate_attention:
            self.agg_attention_weights = nn.Sequential(
                nn.Linear(embedding_dim, 128),  
                nn.ReLU(),
                nn.Linear(128, 1, bias=False)  
            )


    def component_attention_pooling(self, cpt_agg, icd_agg, ttnc_embeds, ttnc_padding_mask):
        components = [cpt_agg, icd_agg, ttnc_embeds]
        # Stack components
        stacked_components = torch.stack(components, dim=2)
        stacked_masks = ttnc_padding_mask.unsqueeze(-1).expand(-1, -1, len(components)).unsqueeze(-1)
        stacked_components = torch.where(stacked_masks, stacked_components, torch.zeros_like(stacked_components))

        if self.use_component_attention:
            # Compute attention scores
            attention_scores = [self.component_attention(comp).squeeze(-1) for comp in components]
            attention_scores = torch.stack(attention_scores, dim=-1)

            # Apply mask
            attention_scores = torch.where(stacked_masks.squeeze(-1), attention_scores, float('-inf'))
            all_inf_mask = torch.isneginf(attention_scores).all(dim=-1)
            if all_inf_mask.any():
                attention_scores = attention_scores.masked_fill(all_inf_mask.unsqueeze(-1), 0.0)

            attention_weights = torch.softmax(attention_scores, dim=-1)
            weighted_components = stacked_components * attention_weights.unsqueeze(-1)
            component_attention_output = weighted_components.sum(dim=2)
        else:
            # Simple sum or mean over components
            component_attention_output = stacked_components.sum(dim=2)
            # Alternatively, use mean
            # component_attention_output = stacked_components.mean(dim=2)

        return component_attention_output



    def attention_pooling_on_aggregates(self, mean_embeds, attention_embeds, variance_embeds, ttnc_padding_mask):
        components = [mean_embeds, attention_embeds]
        if self.use_variance_embeddings:
            components.append(variance_embeds)

        # Stack components
        stacked_components = torch.stack(components, dim=2)
        stacked_masks = ttnc_padding_mask.unsqueeze(-1).expand(-1, -1, len(components)).unsqueeze(-1)
        stacked_components = torch.where(stacked_masks, stacked_components, torch.zeros_like(stacked_components))

        if self.use_aggregate_attention:
            # Compute attention scores
            valid_embeds = [torch.where(ttnc_padding_mask.unsqueeze(-1), comp, torch.zeros_like(comp)) for comp in components]
            attention_scores = [self.component_attention(comp).squeeze(-1) for comp in valid_embeds]
            attention_scores = torch.stack(attention_scores, dim=-1)

            # Apply mask
            attention_scores = torch.where(stacked_masks.squeeze(-1), attention_scores, float('-inf'))
            all_inf_mask = torch.isneginf(attention_scores).all(dim=-1)
            if all_inf_mask.any():
                attention_scores = attention_scores.masked_fill(all_inf_mask.unsqueeze(-1), 0.0)

            attention_weights = torch.softmax(attention_scores, dim=-1)
            weighted_aggs = stacked_components * attention_weights.unsqueeze(-1)
            agg_attention_output = weighted_aggs.sum(dim=2)
        else:
            # Simple sum or mean over components
            agg_attention_output = stacked_components.sum(dim=2)
            # Alternatively, use mean
            # agg_attention_output = stacked_components.mean(dim=2)

        return agg_attention_output



    def code_attention_pooling(self, embeds, padding_mask, tokens, token_rarity_scores):
        # embeds shape: [batch_size, num_claims, num_codes, emb_size]
        # padding_mask shape: [batch_size, num_claims, num_codes]
        # tokens shape: [batch_size, num_claims, num_codes]

        batch_size, num_claims, num_codes, emb_size = embeds.size()
        embeds = embeds.view(-1, num_codes, emb_size)           # [batch_size*num_claims, num_codes, emb_size]
        padding_mask = padding_mask.view(-1, num_codes)         # [batch_size*num_claims, num_codes]
        tokens = tokens.reshape(-1, num_codes)                     # [batch_size*num_claims, num_codes]

        if self.use_code_attention:
            # Compute attention scores
            attention_scores = self.attention_weights(embeds).squeeze(-1)  # [batch_size*num_claims, num_codes]
            if self.use_token_rarity and token_rarity_scores is not None:
                token_rarity_scores = token_rarity_scores[tokens].to(embeds.device)
                # Clamp to prevent log(0)
                token_rarity_scores = torch.clamp(token_rarity_scores, min=1e-8)
                attention_scores += torch.log(token_rarity_scores)
            # Apply mask
            attention_scores = torch.where(padding_mask, attention_scores, float('-inf'))
            # Handle sequences where all tokens are masked
            all_inf_mask = torch.isneginf(attention_scores).all(dim=-1)
            if all_inf_mask.any():
                # Replace attention_scores with zeros for these sequences
                attention_scores = attention_scores.masked_fill(all_inf_mask.unsqueeze(-1), 0.0)
                attention_weights = torch.zeros_like(attention_scores)
            else:
                attention_weights = torch.softmax(attention_scores, dim=-1)
        else:
            # If code attention is not used, use uniform attention weights over non-masked tokens
            attention_weights = torch.ones_like(padding_mask, dtype=embeds.dtype)
            attention_weights = attention_weights * padding_mask  # Zero out masked positions
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
        device = cpt_tokens.device

        cpt_embeds = self.cpt_embedding(cpt_tokens)
        icd_embeds = self.icd_embedding(icd_tokens)
        ttnc_embeds = self.ttnc_embedding(ttnc_tokens)

        # Create padding masks
        cpt_padding_mask = cpt_tokens != self.cpt_embedding.padding_idx
        icd_padding_mask = icd_tokens != self.icd_embedding.padding_idx
        ttnc_padding_mask = ttnc_tokens != self.ttnc_embedding.padding_idx

        if self.use_token_rarity and self.cpt_rarity_scores is not None:
            self.cpt_rarity_scores = self.cpt_rarity_scores.to(device)
        if self.use_token_rarity and self.icd_rarity_scores is not None:
            self.icd_rarity_scores = self.icd_rarity_scores.to(device)

        # Apply code pooling
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

        # Aggregate embeddings with attention pooling
        cpt_agg = self.attention_pooling_on_aggregates(cpt_mean_embeds, cpt_attention_embeds, cpt_variance_embeds, ttnc_padding_mask)
        icd_agg = self.attention_pooling_on_aggregates(icd_mean_embeds, icd_attention_embeds, icd_variance_embeds, ttnc_padding_mask)

        # Component attention pooling
        aggregated_embeddings = self.component_attention_pooling(cpt_agg, icd_agg, ttnc_embeds, ttnc_padding_mask)

        # Normalize
        aggregated_embeddings = self.layer_norm(aggregated_embeddings)

        return aggregated_embeddings


