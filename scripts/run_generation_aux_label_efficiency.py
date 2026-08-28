"""Measure cost-label efficiency for supervised and label-free auxiliary heads."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint, read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from scripts.run_cost_supervision_ablation import stratified_fraction_positions
from scripts.run_frozen_generation_probe import collect_frozen_dataset, resolve_device, write_json
from scripts.run_generation_aux_cost_probe import evaluate_head, fit_head


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cost-label-fractions", nargs="+", type=float,
        default=[0.01, 0.02, 0.05, 0.1, 0.25, 1.0],
    )
    parser.add_argument("--aux-weight", type=float, default=10.0)
    parser.add_argument("--consistency-weight", type=float, default=300.0)
    parser.add_argument("--consistency-noise-std", type=float, default=0.1)
    parser.add_argument("--sigreg-weight", type=float, default=0.2)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = parser.parse_args(argv)
    if any(not 0.0 < value <= 1.0 for value in args.cost_label_fractions):
        parser.error("--cost-label-fractions must be in (0, 1]")
    return args


def build_specs(args):
    return [
        {
            "name": "cost_only", "aux_weight": 0.0, "aux_modalities": ("cpt", "icd", "ttnc"),
            "consistency_weight": 0.0, "sigreg_weight": 0.0,
        },
        {
            "name": "ttnc_only", "aux_weight": args.aux_weight, "aux_modalities": ("ttnc",),
            "consistency_weight": 0.0, "sigreg_weight": 0.0,
        },
        {
            "name": "aligned_all", "aux_weight": args.aux_weight,
            "aux_modalities": ("cpt", "icd", "ttnc"),
            "consistency_weight": 0.0, "sigreg_weight": 0.0,
        },
        {
            "name": "consistency_sigreg", "aux_weight": 0.0,
            "aux_modalities": ("cpt", "icd", "ttnc"),
            "consistency_weight": args.consistency_weight,
            "sigreg_weight": args.sigreg_weight,
        },
    ]


def aggregate(runs):
    grouped = defaultdict(list)
    cost_only = {}
    for row in runs:
        key = (row["cost_label_fraction"], row["ablation"])
        grouped[key].append(row)
        if row["ablation"] == "cost_only":
            cost_only[(row["cost_label_fraction"], row["seed"])] = row

    result = {}
    for (fraction, ablation), rows in grouped.items():
        maes = [row["cost"]["target_probe_mae_dollars"] for row in rows]
        rmses = [row["cost"]["target_probe_rmse_dollars"] for row in rows]
        deltas = [
            row["cost"]["target_probe_mae_dollars"]
            - cost_only[(fraction, row["seed"])]["cost"]["target_probe_mae_dollars"]
            for row in rows
        ]
        result[f"{fraction:g}|{ablation}"] = {
            "num_runs": len(rows),
            "mae_dollars": {
                "mean": float(np.mean(maes)),
                "sample_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            },
            "rmse_dollars": {
                "mean": float(np.mean(rmses)),
                "sample_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            },
            "paired_mae_delta_vs_cost_only": {
                "mean": float(np.mean(deltas)),
                "sample_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "wins": int(sum(delta < 0 for delta in deltas)),
            },
        }
    return result


def main(argv=None):
    args = parse_args(argv)
    specs = build_specs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.accelerator)

    config = copy.deepcopy(read_checkpoint_config(args.checkpoint))
    if config is None:
        raise ValueError("Checkpoint does not contain a saved Config object")
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.evaluation_split = "val"
    config.use_generative_save = False
    config.use_plotting = False
    config.pretrain_diffusion = False
    config.trainer_accelerator = args.accelerator
    config = apply_runtime_config_overrides(config)
    train_dataset, _, val_dataset, _, config, dataset = prepare_data(
        config, requested_eval_split="val"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )
    encoder = load_claims_model_checkpoint(
        HierarchicalClaimsModel, args.checkpoint, config=config, map_location=device
    )
    train_data = collect_frozen_dataset(
        encoder, train_loader, ["next_claim_prediction"], device
    )
    val_data = collect_frozen_dataset(
        encoder, val_loader, ["next_claim_prediction"], device
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_features = train_data["features"]["next_claim_prediction"]
    val_features = val_data["features"]["next_claim_prediction"]
    formulation = getattr(config, "sigreg_formulation", "lejepa_convex")
    num_slices = int(getattr(config, "sigreg_num_slices", 256))
    num_points = int(getattr(config, "sigreg_num_points", 17))
    runs = []
    for fraction in sorted(set(args.cost_label_fractions)):
        for seed in args.seeds:
            positions = stratified_fraction_positions(train_data["cost"], fraction, seed)
            for spec in specs:
                print(
                    f"Training fraction={fraction:g} seed={seed} "
                    f"ablation={spec['name']} labels={len(positions)}"
                )
                head, mean, std, cost_mean, cost_std, history = fit_head(
                    train_features,
                    train_data["cost"],
                    train_data["cpt_ids"],
                    train_data["icd_ids"],
                    train_data["ttnc"],
                    positions,
                    cpt_vocab_size=config.cpt_vocab_size,
                    icd_vocab_size=config.icd_vocab_size,
                    ttnc_vocab_size=config.ttnc_vocab_size,
                    hidden_dim=args.hidden_dim,
                    batch_size=args.head_batch_size,
                    epochs=args.epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    adapter_dim=0,
                    adapter_l2_weight=0.0,
                    linear_heads=False,
                    aux_weight=spec["aux_weight"],
                    aux_modalities=spec["aux_modalities"],
                    aux_label_mode="aligned",
                    consistency_weight=spec["consistency_weight"],
                    consistency_noise_std=args.consistency_noise_std,
                    sigreg_weight=spec["sigreg_weight"],
                    sigreg_num_slices=num_slices,
                    sigreg_num_points=num_points,
                    sigreg_formulation=formulation,
                    seed=seed,
                    device=device,
                )
                cost, generation, geometry = evaluate_head(
                    head, val_features, mean, std, cost_mean, cost_std, val_data, device
                )
                row = {
                    "cost_label_fraction": float(fraction),
                    "seed": int(seed),
                    "ablation": spec["name"],
                    "aux_weight": float(spec["aux_weight"]),
                    "aux_modalities": list(spec["aux_modalities"]),
                    "consistency_weight": float(spec["consistency_weight"]),
                    "sigreg_weight": float(spec["sigreg_weight"]),
                    "cost": cost,
                    "generation": generation,
                    "adapter_geometry": geometry,
                    "training_first": history[0],
                    "training_last": history[-1],
                }
                runs.append(row)
                stem = f"fraction_{fraction:g}__{spec['name']}__seed{seed}"
                write_json(output_dir / f"{stem}.json", row)
                del head

    summary = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_contract_hash": config.data_contract_hash,
            "vocab_hash": config.vocab_hash,
            "cost_label_fractions": sorted(set(args.cost_label_fractions)),
            "generation_label_fraction": 1.0,
            "aux_weight": args.aux_weight,
            "consistency_weight": args.consistency_weight,
            "consistency_noise_std": args.consistency_noise_std,
            "sigreg_weight": args.sigreg_weight,
            "sigreg_formulation": formulation,
            "sigreg_num_slices": num_slices,
            "sigreg_num_points": num_points,
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "seeds": args.seeds,
        },
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
