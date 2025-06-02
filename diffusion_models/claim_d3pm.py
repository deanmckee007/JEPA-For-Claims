import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.distributions.categorical import Categorical

class ClaimD3PM(pl.LightningModule):
    """Discrete diffusion model for single claim generation."""

    def __init__(self, config, vocab_size, condition_dim):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.vocab_size = vocab_size
        self.condition_dim = condition_dim
        self.num_timesteps = getattr(config, "diffusion_steps", 100)
        self.guidance_scale = getattr(config, "guidance_scale", 4.0)

        # Transition noise schedule (flatten-to-uniform). ``alphas`` controls
        # the probability of replacing a token with uniform noise at each time
        # step. Using a 1-D tensor drastically reduces memory usage compared to
        # the previous pre-computed transition tensor of shape ``(T, V, V)``.
        # Cosine noise schedule starting at ~0.05 and rising to ~0.25
        t = torch.arange(self.num_timesteps, dtype=torch.float32)
        self.alphas = nn.Parameter(
            0.25 - 0.20 * torch.cos(0.5 * torch.pi * t / (self.num_timesteps - 1))
        )

        self.token_embed = nn.Embedding(vocab_size, config.embedding_dim)
        self.time_embed = nn.Embedding(self.num_timesteps, config.embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=4,
            dim_feedforward=config.embedding_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.denoiser = nn.TransformerEncoder(encoder_layer, num_layers=4)
        # Causal inter-slot attention layer used to suppress duplicates
        self.slot_attn = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=4,
            dim_feedforward=config.embedding_dim * 4,
            dropout=0.0,
            batch_first=True,
        )
        self.film_fc = nn.Linear(condition_dim, config.embedding_dim * 2)
        self.output_proj = nn.Linear(config.embedding_dim, vocab_size)
        # Learnable default condition used during diffusion pretrain
        self.default_condition = nn.Parameter(torch.zeros(condition_dim))
        self.lr = getattr(config, "lr", 1e-3)

    def load_state_dict(self, state_dict, strict=True):
        # Old checkpoints stored a large ``log_q`` tensor which has been
        # removed. Guard against loading it so older checkpoints remain usable.
        state_dict.pop('log_q', None)
        return super().load_state_dict(state_dict, strict)

    def q_sample(self, x0, t):
        """Corrupt tokens according to D3PM flatten-to-uniform channel."""
        # ``t`` is a tensor of timesteps for each example in the batch.
        alpha_t = self.alphas[t].to(x0.device).unsqueeze(1)  # (batch, 1)
        keep_mask = torch.bernoulli((1 - alpha_t) * torch.ones_like(x0, dtype=torch.float32)).bool()
        noise = torch.randint_like(x0, low=0, high=self.vocab_size)
        return torch.where(keep_mask, x0, noise)

    def denoise(self, x_t, t, condition=None):
        token_emb = self.token_embed(x_t)
        time_emb = self.time_embed(t).unsqueeze(1)
        time_emb = time_emb.expand(-1, x_t.size(1), -1)
        emb = token_emb + time_emb
        if condition is not None:
            gamma, beta = self.film_fc(condition).chunk(2, dim=-1)
            emb = emb * (1 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        h = self.denoiser(emb)
        # Causal mask so slot j attends only to slots < j
        seq_len = h.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=h.device), diagonal=1).bool()
        h = self.slot_attn(h, mask)
        return self.output_proj(h)

    def p_losses(self, x0, t, condition):
        x_t = self.q_sample(x0, t)
        logits = self.denoise(x_t, t, condition)
        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x0.view(-1),
            ignore_index=0,
        )
        return loss

    def forward(self, tokens, condition=None):
        b = tokens.size(0)
        t = torch.randint(0, self.num_timesteps, (b,), device=tokens.device)
        if condition is None:
            condition = self.default_condition.expand(b, -1)
        loss = self.p_losses(tokens, t, condition)
        return loss

    @torch.no_grad()
    def generate_claim(self, condition, seq_len):
        device = self.token_embed.weight.device
        x = torch.randint(0, self.vocab_size, (condition.size(0), seq_len), device=device)
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((condition.size(0),), step, dtype=torch.long, device=device)
            uncond = self.denoise(x, t, None)
            cond = self.denoise(x, t, condition)
            logits = uncond + self.guidance_scale * (cond - uncond)
            x = Categorical(logits=logits).sample()

        dup_rates = []
        for i in range(x.size(0)):
            tokens = x[i].tolist()
            if 0 in tokens:
                tokens = tokens[: tokens.index(0)]
            dup_count = len(tokens) - len(set(tokens))
            dup_rates.append(dup_count / max(len(tokens), 1))
            tokens = list(dict.fromkeys(tokens))
            padded = tokens + [0] * (seq_len - len(tokens))
            x[i] = torch.tensor(padded, device=x.device)

        self.last_dup_rate = float(sum(dup_rates) / len(dup_rates)) if dup_rates else 0.0
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, batch, batch_idx):
        """Standard Lightning training step for diffusion pretraining."""
        cpt, icd, ttnc, _ = batch
        cpt_tokens = cpt[:, -1, :]
        icd_tokens = icd[:, -1, :]
        ttnc_tokens = ttnc[:, -1]
        tokens = torch.cat([cpt_tokens, icd_tokens, ttnc_tokens.unsqueeze(1)], dim=1)
        loss = self.forward(tokens)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # Explicitly log diffusion cross-entropy and perplexity for monitoring
        self.log("diffusion_ce", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("diff_ppl", torch.exp(loss), on_step=True, on_epoch=True, prog_bar=True)
        with torch.no_grad():
            _ = self.generate_claim(self.default_condition.expand(tokens.size(0), -1), seq_len=tokens.size(1))
            self.log("dup_rate", getattr(self, "last_dup_rate", 0.0), on_step=True, on_epoch=True, prog_bar=True)
        return loss
