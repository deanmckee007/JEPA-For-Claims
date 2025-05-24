import torch
import torch.nn as nn

class TopKActivation(nn.Module):
    """Applies a Top-K sparsity operation to the input."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.k >= x.size(1):
            return x

        # Indices of the top-k values by absolute magnitude
        _, topk_indices = torch.topk(x.abs(), self.k, dim=1)

        mask = torch.zeros_like(x, dtype=torch.bool)
        mask.scatter_(1, topk_indices, True)

        # Verify mask selects exactly k indices per sample during debugging
        if __debug__:
            assert torch.all(mask.sum(dim=1) == self.k), "TopKActivation mask incorrect"

        return x * mask.float()


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder with Top-K activation."""

    def __init__(self, input_dim: int, hidden_dim: int, k: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.activation = TopKActivation(k)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)
        hidden = self.activation(hidden)
        return self.decoder(hidden)
