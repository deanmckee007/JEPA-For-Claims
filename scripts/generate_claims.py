"""CLI utility to sample claims from a trained ClaimD3PM model."""

import torch
import argparse
from diffusion_models import ClaimD3PM
from jepa_utils.config import Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to trained checkpoint")
    parser.add_argument("--num", type=int, default=10)
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Model vocabulary size if not stored in checkpoint")
    parser.add_argument("--condition-dim", type=int, default=None,
                        help="Conditioning dimension if not stored in checkpoint")
    args = parser.parse_args()

    try:
        model = ClaimD3PM.load_from_checkpoint(args.checkpoint)
    except TypeError:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        hparams = ckpt.get("hyper_parameters", {})
        config_dict = hparams.get("config", {})
        config = Config()
        if isinstance(config_dict, dict):
            for k, v in config_dict.items():
                setattr(config, k, v)
        vocab_size = args.vocab_size if args.vocab_size is not None else hparams.get("vocab_size")
        condition_dim = args.condition_dim if args.condition_dim is not None else hparams.get("condition_dim")
        model = ClaimD3PM.load_from_checkpoint(
            args.checkpoint,
            config=config,
            vocab_size=vocab_size,
            condition_dim=condition_dim,
        )
    model.eval()
    condition = torch.zeros(args.num, model.condition_dim, device=model.device)
    samples = model.generate_claim(condition, seq_len=5)
    print(samples.tolist())


if __name__ == "__main__":
    main()
