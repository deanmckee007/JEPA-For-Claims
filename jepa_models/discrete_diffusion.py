import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

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

        self.cpt_threshold = getattr(config, "base_cpt_threshold", 0.5)
        self.icd_threshold = getattr(config, "base_icd_threshold", 0.5)
        self.cpt_prob_agg = getattr(config, "cpt_prob_agg", "max")
        self.ttnc_temperature = getattr(config, "ttnc_temperature", 1.0)

        self.model = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim * 6),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 6, self.embedding_dim * 3),
        )

        self.cpt_proj = nn.Linear(
            self.embedding_dim, self.cpt_vocab_size * config.max_cpt_tokens
        )
        self.icd_proj = nn.Linear(
            self.embedding_dim, self.icd_vocab_size * config.max_icd_tokens
        )
        self.ttnc_proj = nn.Linear(self.embedding_dim, self.ttnc_vocab_size)

        self.lr = config.lr

        self.debug_generation = getattr(config, 'debug_generation', False)

        self.condition_dim = condition_dim or getattr(config, "diffusion_condition_dim", 0)
        if self.condition_dim and self.condition_dim > 0:
            self.condition_proj = nn.Linear(self.condition_dim, self.embedding_dim * 3)
        else:
            self.condition_proj = None

        self.max_cpt_tokens = config.max_cpt_tokens
        self.max_icd_tokens = config.max_icd_tokens

    def q_sample(self, tokens, vocab_size, t):
        """Perform a single diffusion step on the provided tokens.

        Padding positions (token==0) are left unchanged to avoid
        learning dynamics dominated by PAD slots.
        """
        prob = self.betas[t].view(tokens.size(0), *([1] * (tokens.dim() - 1)))
        noise = torch.randint(0, vocab_size, tokens.shape, device=tokens.device)
        mask = torch.bernoulli(prob.expand_as(tokens)).bool()
        mask &= tokens != 0
        return torch.where(mask, noise, tokens)

    def aggregate_tokens(self, cpt_tokens, icd_tokens, ttnc_tokens):
        """Aggregate embeddings for CPT, ICD and TTNC tokens."""
        if cpt_tokens.dim() == 1:
            cpt_tokens = cpt_tokens.unsqueeze(1)
        if icd_tokens.dim() == 1:
            icd_tokens = icd_tokens.unsqueeze(1)

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

        # --- Predict logits for each slot independently ---
        cpt_logits = self.cpt_proj(predicted[:, : self.embedding_dim])
        icd_logits = self.icd_proj(predicted[:, self.embedding_dim : 2 * self.embedding_dim])
        ttnc_logits = self.ttnc_proj(predicted[:, 2 * self.embedding_dim :])

        batch_size = cpt_tokens.size(0)
        cpt_logits = cpt_logits.view(batch_size, self.max_cpt_tokens, self.cpt_vocab_size)
        icd_logits = icd_logits.view(batch_size, self.max_icd_tokens, self.icd_vocab_size)

        loss_cpt = F.cross_entropy(
            cpt_logits.reshape(-1, self.cpt_vocab_size),
            cpt_tokens.reshape(-1),
            reduction="none",
        )
        loss_icd = F.cross_entropy(
            icd_logits.reshape(-1, self.icd_vocab_size),
            icd_tokens.reshape(-1),
            reduction="none",
        )
        loss_ttnc = F.cross_entropy(ttnc_logits, ttnc_tokens, reduction="none")

        cpt_mask = cpt_tokens.reshape(-1) != 0
        icd_mask = icd_tokens.reshape(-1) != 0
        ttnc_mask = ttnc_tokens != 0

        loss_cpt = (loss_cpt * cpt_mask).sum() / (cpt_mask.sum() + 1e-8)
        loss_icd = (loss_icd * icd_mask).sum() / (icd_mask.sum() + 1e-8)
        loss_ttnc = (loss_ttnc * ttnc_mask).sum() / (ttnc_mask.sum() + 1e-8)
        loss = loss_cpt + loss_icd + loss_ttnc
        if self.debug_generation and t.numel() == 1:
            print(f"t={t.item()}  loss_mean={loss.item():.4f}")
        return loss

    def forward(self, cpt_tokens, icd_tokens, ttnc_tokens, condition=None):
        batch_size = cpt_tokens.size(0)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=cpt_tokens.device).long()
        loss = self.p_losses(cpt_tokens, icd_tokens, ttnc_tokens, t, condition)
        return loss

    @torch.no_grad()
    def sample(self, batch_size, condition=None):
        device = self.betas.device
        cpt_tokens = torch.randint(
            0, self.cpt_vocab_size, (batch_size, self.max_cpt_tokens), device=device
        )
        icd_tokens = torch.randint(
            0, self.icd_vocab_size, (batch_size, self.max_icd_tokens), device=device
        )
        ttnc_tokens = torch.randint(0, self.ttnc_vocab_size, (batch_size,), device=device)

        # Tensors to hold diagnostics from the final denoising step
        cpt_entropy = torch.zeros(batch_size, device=device)
        icd_entropy = torch.zeros(batch_size, device=device)
        dynamic_cpt_threshold = torch.zeros(batch_size, device=device)
        dynamic_icd_threshold = torch.zeros(batch_size, device=device)

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

            batch_size = cpt_tokens.size(0)
            cpt_logits = cpt_logits.view(batch_size, self.max_cpt_tokens, self.cpt_vocab_size)
            icd_logits = icd_logits.view(batch_size, self.max_icd_tokens, self.icd_vocab_size)

            if step > 0:
                cpt_tokens = torch.argmax(cpt_logits, dim=-1)
                icd_tokens = torch.argmax(icd_logits, dim=-1)
                ttnc_tokens = torch.argmax(ttnc_logits, dim=-1)
                cpt_tokens = self._deduplicate(cpt_tokens)
                icd_tokens = self._deduplicate(icd_tokens)

                if self.debug_generation and step % max(self.num_timesteps // 3, 1) == 0:
                    print(
                        f"diffusion step {step}: CPT[0]={cpt_tokens[0].tolist()} ICD[0]={icd_tokens[0].tolist()} TTNC[0]={ttnc_tokens[0].item()}"
                    )
            else:
                cpt_probs_raw = torch.sigmoid(cpt_logits)
                icd_probs_raw = torch.sigmoid(icd_logits)
                if self.cpt_prob_agg == "mean":
                    cpt_probs = cpt_probs_raw.mean(dim=1)
                    icd_probs = icd_probs_raw.mean(dim=1)
                elif self.cpt_prob_agg == "sum":
                    cpt_probs = cpt_probs_raw.sum(dim=1)
                    icd_probs = icd_probs_raw.sum(dim=1)
                else:
                    cpt_probs = cpt_probs_raw.max(dim=1).values
                    icd_probs = icd_probs_raw.max(dim=1).values
                ttnc_probs = torch.softmax(ttnc_logits / self.ttnc_temperature, dim=-1)

                cpt_entropy = -torch.sum(
                    cpt_probs.clamp(1e-8, 1 - 1e-8)
                    * torch.log(cpt_probs.clamp(1e-8, 1 - 1e-8)),
                    dim=1,
                )
                icd_entropy = -torch.sum(
                    icd_probs.clamp(1e-8, 1 - 1e-8)
                    * torch.log(icd_probs.clamp(1e-8, 1 - 1e-8)),
                    dim=1,
                )
                cpt_norm = cpt_entropy / math.log(cpt_probs.size(-1))
                icd_norm = icd_entropy / math.log(icd_probs.size(-1))
                cpt_thresh = (self.cpt_threshold * (1.0 - cpt_norm)).clamp(min=0.1)
                icd_thresh = (self.icd_threshold * (1.0 - icd_norm)).clamp(min=0.1)

                # store diagnostics
                dynamic_cpt_threshold = cpt_thresh
                dynamic_icd_threshold = icd_thresh

                cpt_tokens = (cpt_probs > cpt_thresh.unsqueeze(-1)).long()
                icd_tokens = (icd_probs > icd_thresh.unsqueeze(-1)).long()
                cpt_tokens = self._deduplicate(cpt_tokens)
                icd_tokens = self._deduplicate(icd_tokens)
                ttnc_tokens = torch.multinomial(ttnc_probs, 1).squeeze(1)

                if self.debug_generation:
                    print(
                        f"final stats CPT p[min={cpt_probs.min():.3f} max={cpt_probs.max():.3f} mean={cpt_probs.mean():.3f}]"
                    )
                    print(
                        f"final stats ICD p[min={icd_probs.min():.3f} max={icd_probs.max():.3f} mean={icd_probs.mean():.3f}]"
                    )
                    print(f"TTNC softmax[0]={ttnc_probs[0].tolist()}")
                    print(
                        f"denoised emb var={h.var().item():.3f} min={h.min().item():.3f} max={h.max().item():.3f}"
                    )


        return {
            'cpt_tokens': cpt_tokens,
            'icd_tokens': icd_tokens,
            'ttnc_token': ttnc_tokens,
            'cpt_entropy': cpt_entropy,
            'cpt_threshold': dynamic_cpt_threshold,
            'icd_entropy': icd_entropy,
            'icd_threshold': dynamic_icd_threshold,
        }

    def _deduplicate(self, tokens, pad_value=0):
        """Remove duplicate codes within each claim and pad the rest."""
        unique_tokens = []
        for claim in tokens.tolist():
            seen = set()
            filtered = []
            for tok in claim:
                if tok not in seen:
                    filtered.append(tok)
                    seen.add(tok)
            filtered += [pad_value] * (len(claim) - len(filtered))
            unique_tokens.append(filtered)
        return tokens.new_tensor(unique_tokens)

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
