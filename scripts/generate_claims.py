"""CLI utility to sample claims from a trained ClaimD3PM model."""

import torch
import argparse
from jepa_models.diffusion_models.claim_d3pm import ClaimD3PM


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to trained checkpoint")
    parser.add_argument("--num", type=int, default=10)
    args = parser.parse_args()

    model = ClaimD3PM.load_from_checkpoint(args.checkpoint)
    model.eval()
    condition = torch.zeros(args.num, model.condition_dim, device=model.device)
    samples = model.generate_claim(condition, seq_len=5)
    print(samples.tolist())


if __name__ == "__main__":
    main()
