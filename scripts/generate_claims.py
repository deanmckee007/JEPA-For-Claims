"""CLI utility to sample claims from a trained ClaimD3PM model."""

import torch
import argparse
from diffusion_models import ClaimD3PM
from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to trained checkpoint")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Model vocabulary size if not stored in checkpoint (legacy)",
    )
    parser.add_argument(
        "--condition-dim",
        type=int,
        default=None,
        help="Conditioning dimension if not stored in checkpoint (legacy)",
    )
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    config_dict = hparams.get("config", {})
    config = Config()
    if isinstance(config_dict, dict):
        for k, v in config_dict.items():
            setattr(config, k, v)

    try:
        model = HierarchicalClaimsModel.load_from_checkpoint(
            args.checkpoint, config=config, strict=False
        )
    except TypeError:
        # Fallback for very old checkpoints containing only the diffusion model
        vocab_size = args.vocab_size or hparams.get("vocab_size")
        if vocab_size is None:
            vocab_size = (
                getattr(config, "cpt_vocab_size", 0)
                + getattr(config, "icd_vocab_size", 0)
                + getattr(config, "ttnc_vocab_size", 0)
                + 3
            )
        condition_dim = args.condition_dim or hparams.get("condition_dim")
        if condition_dim is None:
            condition_dim = getattr(
                config, "output_dim", getattr(config, "embedding_dim", 128)
            )
        model = ClaimD3PM.load_from_checkpoint(
            args.checkpoint,
            config=config,
            vocab_size=vocab_size,
            condition_dim=condition_dim,
        )
        diffusion_only = True
    else:
        diffusion_only = False

    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if diffusion_only:
        cond_dim = model.condition_dim
        diffusion_model = model
    else:
        cond_dim = model.config.output_dim
        diffusion_model = model.diffusion_model

    seq_len = getattr(config, "max_cpt_tokens", 0) + getattr(config, "max_icd_tokens", 0) + 1
    condition = torch.zeros(args.num, cond_dim, device=device)
    samples = diffusion_model.generate_claim(condition, seq_len=seq_len)
    print(samples.tolist())


if __name__ == "__main__":
    main()
