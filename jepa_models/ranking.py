import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseCostRankingHead(nn.Module):
    """Small score head trained only through pairwise cost orderings."""

    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, representations: torch.Tensor) -> torch.Tensor:
        return self.network(representations).squeeze(-1)


def pairwise_ranknet_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    partner_indices: torch.Tensor,
) -> torch.Tensor:
    """RankNet logistic loss over a deterministic set of within-batch pairs."""
    target_difference = targets - targets.index_select(0, partner_indices)
    valid = target_difference.ne(0)
    if not valid.any():
        return scores.sum() * 0.0
    score_difference = scores - scores.index_select(0, partner_indices)
    labels = target_difference.gt(0).to(score_difference.dtype)
    return F.binary_cross_entropy_with_logits(
        score_difference[valid],
        labels[valid],
    )


def all_pair_indices(num_items: int, device=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return every unique unordered pair in a group."""
    if num_items < 2:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.triu_indices(num_items, num_items, offset=1, device=device)


def pairwise_logistic_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    *,
    pair_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """RankNet logistic loss over explicit pairs with optional Lambda weights."""
    target_difference = (
        targets.index_select(0, left_indices)
        - targets.index_select(0, right_indices)
    )
    valid = target_difference.ne(0)
    if not valid.any():
        return scores.sum() * 0.0

    score_difference = (
        scores.index_select(0, left_indices)
        - scores.index_select(0, right_indices)
    )
    labels = target_difference.gt(0).to(score_difference.dtype)
    losses = F.binary_cross_entropy_with_logits(
        score_difference[valid],
        labels[valid],
        reduction="none",
    )
    if pair_weights is not None:
        weights = pair_weights[valid].detach().to(losses.dtype)
        positive = weights > 0
        if not positive.any():
            return losses.mean()
        weights = weights / weights[positive].mean().clamp_min(1e-8)
        losses = losses * weights
    return losses.mean()


def lambdarank_pair_weights(
    scores: torch.Tensor,
    relevance: torch.Tensor,
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    *,
    gain_scale: float = 4.0,
) -> torch.Tensor:
    """Compute detached absolute delta-NDCG weights for proposed pair swaps.

    ``relevance`` is expected to be an empirical percentile in [0, 1]. Scaling
    percentiles instead of raw dollar values prevents exponential DCG gains
    from being dominated by a handful of extreme claims.
    """
    if left_indices.numel() == 0:
        return scores.new_empty((0,))

    with torch.no_grad():
        num_items = scores.numel()
        predicted_order = torch.argsort(scores, descending=True)
        predicted_rank = torch.empty_like(predicted_order)
        predicted_rank[predicted_order] = torch.arange(
            num_items,
            device=scores.device,
        )
        discounts = 1.0 / torch.log2(predicted_rank.to(scores.dtype) + 2.0)

        gains = torch.pow(2.0, relevance.to(scores.dtype) * gain_scale) - 1.0
        ideal_order = torch.argsort(relevance, descending=True)
        ideal_discounts = 1.0 / torch.log2(
            torch.arange(num_items, device=scores.device, dtype=scores.dtype) + 2.0
        )
        ideal_dcg = (
            gains.index_select(0, ideal_order) * ideal_discounts
        ).sum().clamp_min(1e-8)

        gain_delta = torch.abs(
            gains.index_select(0, left_indices)
            - gains.index_select(0, right_indices)
        )
        discount_delta = torch.abs(
            discounts.index_select(0, left_indices)
            - discounts.index_select(0, right_indices)
        )
        return gain_delta * discount_delta / ideal_dcg
