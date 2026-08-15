import torch
import torch.nn as nn


class ClaimsWorldModel(nn.Module):
    """Deterministic latent dynamics over claim-level sequence representations.

    The module consumes encoded context-claim latents and rolls a compact patient
    state forward in latent space. It is intentionally lightweight: no token
    reconstruction and no action-conditioned planning, just a claims-native
    recurrent state with imagined future claim latents.
    """

    def __init__(
        self,
        claim_dim: int,
        state_dim: int,
        future_steps: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.claim_dim = claim_dim
        self.state_dim = state_dim
        self.future_steps = future_steps

        self.input_proj = nn.Sequential(
            nn.LayerNorm(claim_dim),
            nn.Linear(claim_dim, state_dim),
            nn.GELU(),
        )
        self.encoder = nn.GRU(
            input_size=state_dim,
            hidden_size=state_dim,
            num_layers=1,
            batch_first=True,
        )
        self.transition = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, state_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(state_dim, state_dim),
        )
        self.claim_decoder = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, claim_dim),
        )

    def _last_valid_state(self, encoded_sequence: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        valid_lengths = valid_mask.long().sum(dim=1).clamp(min=1)
        batch_indices = torch.arange(encoded_sequence.size(0), device=encoded_sequence.device)
        last_indices = valid_lengths - 1
        return encoded_sequence[batch_indices, last_indices]

    def forward(self, context_claims: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        projected = self.input_proj(context_claims)
        encoded_sequence, _ = self.encoder(projected)
        state = self._last_valid_state(encoded_sequence, valid_mask)

        rollout_states = []
        rollout_claims = []
        current_state = state
        for _ in range(self.future_steps):
            current_state = current_state + self.transition(current_state)
            rollout_states.append(current_state)
            rollout_claims.append(self.claim_decoder(current_state))

        rollout_states = torch.stack(rollout_states, dim=1)
        rollout_claims = torch.stack(rollout_claims, dim=1)

        return {
            "world_model_state": state,
            "world_model_rollout_states": rollout_states,
            "world_model_rollout_claims": rollout_claims,
            "world_model_next_claim": rollout_claims[:, 0],
            "world_model_future_summary": rollout_claims.mean(dim=1),
        }
