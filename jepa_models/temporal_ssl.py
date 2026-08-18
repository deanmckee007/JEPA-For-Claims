import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalContrastiveHead(nn.Module):
    """Claims-native temporal contrastive objective over context and future latents."""

    def __init__(
        self,
        context_dim: int,
        future_dim: int,
        projection_dim: int,
        temperature: float = 0.1,
        mode: str = "cpc",
        context_k: int = 4,
    ):
        super().__init__()
        self.temperature = temperature
        self.mode = mode
        self.context_k = context_k
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.future_proj = nn.Sequential(
            nn.LayerNorm(future_dim),
            nn.Linear(future_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).float()
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
        return (values * weights).sum(dim=1) / counts

    def _recent_mask(self, mask: torch.Tensor) -> torch.Tensor:
        if self.context_k <= 0:
            return mask

        recent_mask = torch.zeros_like(mask)
        for batch_index in range(mask.size(0)):
            valid_indices = torch.nonzero(mask[batch_index], as_tuple=False).squeeze(1)
            if valid_indices.numel() == 0:
                continue
            selected = valid_indices[-self.context_k :]
            recent_mask[batch_index, selected] = True
        return recent_mask

    def build_context_view(self, sequence_output: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        if self.mode == "cpc":
            pooled_mask = self._recent_mask(context_mask)
        else:
            pooled_mask = context_mask
        return self._masked_mean(sequence_output, pooled_mask)

    def build_future_view(self, future_targets: torch.Tensor, future_mask: torch.Tensor) -> torch.Tensor:
        return self._masked_mean(future_targets, future_mask)

    def forward(
        self,
        sequence_output: torch.Tensor,
        context_mask: torch.Tensor,
        future_targets: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        zero = sequence_output.new_zeros(())
        valid_examples = context_mask.any(dim=1) & future_mask.any(dim=1)
        if valid_examples.sum() < 2:
            empty_context = sequence_output.new_zeros((0, self.context_proj[-1].out_features))
            empty_future = sequence_output.new_zeros((0, self.future_proj[-1].out_features))
            return {
                "loss": zero,
                "context_view": empty_context,
                "future_view": empty_future,
                "valid_examples": valid_examples,
            }

        context_view = self.build_context_view(sequence_output, context_mask)[valid_examples]
        future_view = self.build_future_view(future_targets, future_mask)[valid_examples]

        projected_context = F.normalize(self.context_proj(context_view), dim=-1)
        projected_future = F.normalize(self.future_proj(future_view), dim=-1)

        logits = projected_context @ projected_future.T
        logits = logits / max(self.temperature, 1e-6)
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.T, labels)
        )
        return {
            "loss": loss,
            "context_view": projected_context,
            "future_view": projected_future,
            "valid_examples": valid_examples,
        }
