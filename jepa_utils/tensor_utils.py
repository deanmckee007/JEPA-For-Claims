# utils/tensor_utils.py
import torch
import random
import math

def masked_mean(tensor, mask, dim):
    """
    Computes the masked mean of a tensor along a specified dimension.
    
    Args:
        tensor (torch.Tensor): The input tensor.
        mask (torch.Tensor): The mask tensor indicating valid elements.
        dim (int): The dimension along which to compute the mean.
    
    Returns:
        torch.Tensor: The masked mean.
    """
    mask = mask.float()  # Convert mask to float
    masked_sum = (tensor * mask.unsqueeze(-1)).sum(dim=dim)
    valid_counts = mask.sum(dim=dim).clamp(min=1)
    mean = masked_sum / valid_counts.unsqueeze(-1)
    mean = mean * (valid_counts.unsqueeze(-1) > 0).float()
    return mean


def masked_variance(tensor, mask, dim):
    # Compute valid counts
    valid_counts = mask.sum(dim=dim, keepdim=True).clamp(min=1)  # Shape: [batch_size, num_claims, 1]

    # Compute masked mean
    mean = masked_mean(tensor, mask, dim=dim)  # Shape: [batch_size, num_claims, emb_size]
    mean = mean.unsqueeze(dim)  # Shape: [batch_size, num_claims, 1, emb_size]

    # Expand mask for broadcasting
    mask_expanded = mask.unsqueeze(-1)  # Shape: [batch_size, num_claims, num_codes, 1]

    # Compute differences
    diff = (tensor - mean) * mask_expanded  # Shape: [batch_size, num_claims, num_codes, emb_size]

    # Compute squared differences
    squared_diff = diff ** 2  # Same shape as `diff`

    # Sum over the specified dimension
    variance_numerator = squared_diff.sum(dim=dim)  # Shape: [batch_size, num_claims, emb_size]

    # Divide by valid counts
    variance = variance_numerator / valid_counts  # Shape: [batch_size, num_claims, emb_size]

    # Handle cases where valid counts <= 1
    mask_valid = (valid_counts > 1)  # Shape: [batch_size, num_claims, 1]
    mask_valid = mask_valid.expand(-1, -1, variance.size(-1))  # Expand to [batch_size, num_claims, emb_size]
    variance = torch.where(mask_valid, variance, torch.zeros_like(variance))

    return variance

def calculate_entropy(probs):
    """Compute the Shannon entropy for a probability distribution."""
    # Clamp to avoid log(0)
    p = probs.clamp(min=1e-8, max=1 - 1e-8)
    log_p = torch.log(p)
    entropy = -torch.sum(p * log_p, dim=-1)
    return entropy

def normalized_entropy(probs):
    """Entropy normalised by the maximum possible entropy for the vocabulary."""
    entropy = calculate_entropy(probs)
    vocab_size = probs.size(-1)
    return entropy / math.log(vocab_size)

def adaptive_sampling(probs, entropy, base_temp=1.0, base_top_p=0.9, entropy_adjustment_factor=0.5):
    random.seed()
    # Adjust temperature based on entropy
    temperature = base_temp / (1 + entropy_adjustment_factor * entropy.unsqueeze(-1))
    # Ensure temperature is within a reasonable range
    temperature = torch.clamp(temperature, min=0.5, max=2.0)

    # Apply temperature
    adjusted_logits = torch.log(probs + 1e-12) / temperature

    # Apply top-p filtering
    sorted_probs, sorted_indices = torch.sort(torch.softmax(adjusted_logits, dim=-1), descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = cumulative_probs > base_top_p
    cutoff[..., 0] = False  # Ensure at least one token remains
    sorted_probs[cutoff] = 0
    sorted_probs = sorted_probs / torch.sum(sorted_probs, dim=-1, keepdim=True)

    # Sample from adjusted distribution
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_token)
    return next_token




