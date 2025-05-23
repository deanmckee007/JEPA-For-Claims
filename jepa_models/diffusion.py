import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class DiffusionModel(pl.LightningModule):
    """Simple DDPM for CPT/ICD/TTNC token generation.

    Optionally conditions generation on an external representation when
    ``condition_dim`` is provided.
    """

    def __init__(self, config, condition_dim=None):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_dim = config.embedding_dim
        self.cpt_vocab_size = config.cpt_vocab_size
        self.icd_vocab_size = config.icd_vocab_size
        self.ttnc_vocab_size = config.ttnc_vocab_size
        self.num_timesteps = getattr(config, 'diffusion_steps', 100)

        betas = torch.linspace(1e-4, 0.02, self.num_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))

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

        # Optional conditioning for downstream tasks
        self.condition_dim = condition_dim or getattr(config, 'diffusion_condition_dim', 0)
        if self.condition_dim and self.condition_dim > 0:
            self.condition_proj = nn.Linear(self.condition_dim, self.embedding_dim * 3)
        else:
            self.condition_proj = None

    def q_sample(self, x_start, t, noise):
        sqrt_acp = self.sqrt_alphas_cumprod[t].unsqueeze(1)
        sqrt_om_acp = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
        return sqrt_acp * x_start + sqrt_om_acp * noise

    def p_losses(self, x_start, t, noise, condition=None):
        x_noisy = self.q_sample(x_start, t, noise)
        t_emb = self.time_embed(t)
        if condition is not None and self.condition_proj is not None:
            cond = self.condition_proj(condition)
        else:
            cond = 0
        predicted = self.model(x_noisy + t_emb + cond)
        return F.mse_loss(predicted, noise)

    def aggregate_tokens(self, cpt_tokens, icd_tokens, ttnc_tokens):
        """Aggregate token embeddings into a single representation."""
        # ``cpt_tokens`` and ``icd_tokens`` may come in as 1D tensors when only
        # a single code per claim is used. Ensure we always have a sequence
        # dimension so ``sum(dim=1)`` works for both cases.
        if cpt_tokens.dim() == 1:
            cpt_tokens = cpt_tokens.unsqueeze(1)
        if icd_tokens.dim() == 1:
            icd_tokens = icd_tokens.unsqueeze(1)

        cpt_emb = self.cpt_embedding(cpt_tokens).sum(dim=1)
        icd_emb = self.icd_embedding(icd_tokens).sum(dim=1)
        ttnc_emb = self.ttnc_embedding(ttnc_tokens)
        return torch.cat([cpt_emb, icd_emb, ttnc_emb], dim=1)

    def forward(self, cpt_tokens, icd_tokens, ttnc_tokens, condition=None):
        x_start = self.aggregate_tokens(cpt_tokens, icd_tokens, ttnc_tokens)
        batch_size = x_start.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_start.device).long()
        noise = torch.randn_like(x_start)
        loss = self.p_losses(x_start, t, noise, condition)
        return loss

    @torch.no_grad()
    def sample(self, batch_size, condition=None):
        device = self.betas.device
        x = torch.randn(batch_size, self.embedding_dim * 3, device=device)
        if condition is not None and self.condition_proj is not None:
            cond = self.condition_proj(condition)
        else:
            cond = 0
        for step in reversed(range(self.num_timesteps)):
            t = torch.full((batch_size,), step, device=device, dtype=torch.long)
            t_emb = self.time_embed(t)
            noise_pred = self.model(x + t_emb + cond)
            coef1 = 1 / torch.sqrt(1 - self.betas[step])
            coef2 = self.betas[step] / torch.sqrt(1 - self.alphas_cumprod[step])
            x = coef1 * (x - coef2 * noise_pred)
            if step > 0:
                noise = torch.randn_like(x)
                sigma = torch.sqrt(self.betas[step])
                x += sigma * noise

        cpt_logits = self.cpt_proj(x[:, :self.embedding_dim])
        icd_logits = self.icd_proj(x[:, self.embedding_dim:2 * self.embedding_dim])
        ttnc_logits = self.ttnc_proj(x[:, 2 * self.embedding_dim:])
        cpt_tokens = torch.argmax(cpt_logits, dim=-1)
        icd_tokens = torch.argmax(icd_logits, dim=-1)
        ttnc_token = torch.argmax(ttnc_logits, dim=-1)
        return cpt_tokens, icd_tokens, ttnc_token

    def training_step(self, batch, batch_idx):
        cpt_tensor, icd_tensor, ttnc_tensor, _ = batch
        cpt_tokens = cpt_tensor[:, -1, :]
        icd_tokens = icd_tensor[:, -1, :]
        ttnc_tokens = ttnc_tensor[:, -1]
        loss = self.forward(cpt_tokens, icd_tokens, ttnc_tokens)
        self.log('diffusion_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
