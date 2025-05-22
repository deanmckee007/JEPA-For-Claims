import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DiscreteDiffusionModel(pl.LightningModule):
    """Simplified discrete diffusion model for token generation."""

    def __init__(self, config, condition_dim=None):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_dim = config.embedding_dim
        self.cpt_vocab_size = config.cpt_vocab_size
        self.icd_vocab_size = config.icd_vocab_size
        self.ttnc_vocab_size = config.ttnc_vocab_size
        self.num_timesteps = getattr(config, "diffusion_steps", 10)

        betas = torch.linspace(0.1, 0.9, self.num_timesteps)
        self.register_buffer("betas", betas)

        self.time_embed = nn.Embedding(self.num_timesteps, self.embedding_dim * 3)

        self.cpt_embedding = nn.Embedding(self.cpt_vocab_size, self.embedding_dim)
        self.icd_embedding = nn.Embedding(self.icd_vocab_size, self.embedding_dim)
        self.ttnc_embedding = nn.Embedding(self.ttnc_vocab_size, self.embedding_dim)

        self.model = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim * 6),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 6, self.embedding_dim * 3),
        )

        self.cpt_proj = nn.Linear(self.embedding_dim, self.cpt_vocab_size)
        self.icd_proj = nn.Linear(self.embedding_dim, self.icd_vocab_size)
        self.ttnc_proj = nn.Linear(self.embedding_dim, self.ttnc_vocab_size)

        self.lr = config.lr

        self.condition_dim = condition_dim or getattr(config, "diffusion_condition_dim", 0)
        if self.condition_dim and self.condition_dim > 0:
            self.condition_proj = nn.Linear(self.condition_dim, self.embedding_dim * 3)
        else:
            self.condition_proj = None

        self.max_cpt_tokens = config.max_cpt_tokens
        self.max_icd_tokens = config.max_icd_tokens

    def q_sample(self, tokens, vocab_size, t):
        prob = self.betas[t].unsqueeze(1)
        noise = torch.randint(0, vocab_size, tokens.shape, device=tokens.device)
        mask = torch.bernoulli(prob.expand_as(tokens)).bool()
        return torch.where(mask, noise, tokens)

    def aggregate_tokens(self, cpt_tokens, icd_tokens, ttnc_tokens):
        cpt_emb = self.cpt_embedding(cpt_tokens).sum(dim=1)
        icd_emb = self.icd_embedding(icd_tokens).sum(dim=1)
        ttnc_emb = self.ttnc_embedding(ttnc_tokens)
        return torch.cat([cpt_emb, icd_emb, ttnc_emb], dim=1)

    def p_losses(self, cpt_tokens, icd_tokens, ttnc_tokens, t, condition=None):
        cpt_noisy = self.q_sample(cpt_tokens, self.cpt_vocab_size, t)
        icd_noisy = self.q_sample(icd_tokens, self.icd_vocab_size, t)
        ttnc_noisy = self.q_sample(ttnc_tokens.unsqueeze(1), self.ttnc_vocab_size, t).squeeze(1)
        x_noisy = self.aggregate_tokens(cpt_noisy, icd_noisy, ttnc_noisy)
        t_emb = self.time_embed(t)
        if condition is not None and self.condition_proj is not None:
            cond = self.condition_proj(condition)
        else:
            cond = 0
        predicted = self.model(x_noisy + t_emb + cond)
        cpt_logits = self.cpt_proj(predicted[:, : self.embedding_dim])
        icd_logits = self.icd_proj(predicted[:, self.embedding_dim : 2 * self.embedding_dim])
        ttnc_logits = self.ttnc_proj(predicted[:, 2 * self.embedding_dim :])
        cpt_target = cpt_tokens[:, 0]
        icd_target = icd_tokens[:, 0]
        loss = (
            F.cross_entropy(cpt_logits, cpt_target)
            + F.cross_entropy(icd_logits, icd_target)
            + F.cross_entropy(ttnc_logits, ttnc_tokens)
        )
        return loss

    def forward(self, cpt_tokens, icd_tokens, ttnc_tokens, condition=None):
        batch_size = cpt_tokens.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=cpt_tokens.device).long()
        loss = self.p_losses(cpt_tokens, icd_tokens, ttnc_tokens, t, condition)
        return loss

    @torch.no_grad()
    def sample(self, batch_size, condition=None):
        device = self.betas.device
        cpt_tokens = torch.randint(0, self.cpt_vocab_size, (batch_size, self.max_cpt_tokens), device=device)
        icd_tokens = torch.randint(0, self.icd_vocab_size, (batch_size, self.max_icd_tokens), device=device)
        ttnc_tokens = torch.randint(0, self.ttnc_vocab_size, (batch_size,), device=device)
        if condition is not None and self.condition_proj is not None:
            cond = self.condition_proj(condition)
        else:
            cond = 0
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            x = self.aggregate_tokens(cpt_tokens, icd_tokens, ttnc_tokens)
            t_emb = self.time_embed(t)
            h = self.model(x + t_emb + cond)
            cpt_logits = self.cpt_proj(h[:, : self.embedding_dim])
            icd_logits = self.icd_proj(h[:, self.embedding_dim : 2 * self.embedding_dim])
            ttnc_logits = self.ttnc_proj(h[:, 2 * self.embedding_dim :])
            cpt_tokens = torch.argmax(cpt_logits, dim=-1).unsqueeze(1).repeat(1, self.max_cpt_tokens)
            icd_tokens = torch.argmax(icd_logits, dim=-1).unsqueeze(1).repeat(1, self.max_icd_tokens)
            ttnc_tokens = torch.argmax(ttnc_logits, dim=-1)
        return cpt_tokens[:, 0], icd_tokens[:, 0], ttnc_tokens

    def training_step(self, batch, batch_idx):
        cpt_tensor, icd_tensor, ttnc_tensor, _ = batch
        cpt_tokens = cpt_tensor[:, -1, :]
        icd_tokens = icd_tensor[:, -1, :]
        ttnc_tokens = ttnc_tensor[:, -1]
        loss = self.forward(cpt_tokens, icd_tokens, ttnc_tokens)
        self.log("diffusion_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
