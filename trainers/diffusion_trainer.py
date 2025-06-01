"""Utility to pretrain ClaimD3PM."""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from jepa_models.diffusion_models.claim_d3pm import ClaimD3PM


def run_pretrain(cfg, dataloader: DataLoader):
    model = ClaimD3PM(
        cfg,
        cfg.cpt_vocab_size + cfg.icd_vocab_size + cfg.ttnc_vocab_size + 3,
        cfg.output_dim,
    )
    trainer = pl.Trainer(
        max_epochs=getattr(cfg, "pretrain_diffusion_epochs", 20),
        accelerator="gpu" if pl.utilities.device_parser.num_cuda_devices() > 0 else "cpu",
        log_every_n_steps=10,
    )
    trainer.fit(model, dataloader)
    return model, trainer
