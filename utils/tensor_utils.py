# utils/tensor_utils.py
import torch

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


