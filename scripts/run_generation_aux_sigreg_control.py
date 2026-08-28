"""Test repository-native SIGReg against collapse in label-free consistency training."""

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
    parser.add_argument("--cost-label-fraction", type=float, default=0.1)
    parser.add_argument("--aux-weight", type=float, default=10.0)
    parser.add_argument("--consistency-weight", type=float, default=300.0)
    parser.add_argument("--consistency-noise-std", type=float, default=0.1)
    parser.add_argument("--sigreg-weights", nargs="+", type=float, default=[0.01, 0.05])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--conditions", nargs="+", choices=["pretrained", "raw_history"],
        default=["pretrained"],
    )
    parser.add_argument("--skip-reference-arms", action="store_true")
    return parser.parse_args(argv)


def build_specs(args):
    specs = [] if args.skip_reference_arms else [
        {"name": "cost_only", "aux_weight": 0.0, "label_mode": "aligned", "consistency": 0.0, "sigreg": 0.0},
        {"name": "aligned_all", "aux_weight": args.aux_weight, "label_mode": "aligned", "consistency": 0.0, "sigreg": 0.0},
        {"name": "shuffled_all", "aux_weight": args.aux_weight, "label_mode": "shuffled", "consistency": 0.0, "sigreg": 0.0},
        {"name": f"consistency_{args.consistency_weight:g}", "aux_weight": 0.0, "label_mode": "aligned", "consistency": args.consistency_weight, "sigreg": 0.0},
    ]
    for weight in args.sigreg_weights:
        specs.extend([
            {
                "name": f"sigreg_{weight:g}", "aux_weight": 0.0,
                "label_mode": "aligned", "consistency": 0.0, "sigreg": float(weight),
            },
            {
                "name": f"consistency_{args.consistency_weight:g}_sigreg_{weight:g}",
                "aux_weight": 0.0, "label_mode": "aligned",
                "consistency": args.consistency_weight, "sigreg": float(weight),
            },
        ])
    return specs


def aggregate(runs):
    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["condition"], row["ablation"])].append(row)
    result = {}
    for (condition, ablation), rows in grouped.items():
        metrics = {}
        paths = {
            "mae_dollars": ("cost", "target_probe_mae_dollars"),
            "rmse_dollars": ("cost", "target_probe_rmse_dollars"),
            "hidden_feature_std": ("adapter_geometry", "shared_hidden_mean_feature_std"),
            "hidden_norm": ("adapter_geometry", "shared_hidden_mean_norm"),
        }
        for name, (section, field) in paths.items():
            values = [row[section][field] for row in rows]
            metrics[name] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        result[f"{condition}|{ablation}"] = {"num_runs": len(rows), "metrics": metrics}
    return result


def main(argv=None):
    args = parse_args(argv)
    specs = build_specs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = output_dir / "heads"
    artifact_dir.mkdir(exist_ok=True)
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
    sources = ["next_claim_prediction", "raw_history_hash"]
    train_data = collect_frozen_dataset(encoder, train_loader, sources, device)
    val_data = collect_frozen_dataset(encoder, val_loader, sources, device)
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    source_for_condition = {
        "pretrained": "next_claim_prediction",
        "raw_history": "raw_history_hash",
    }
    formulation = getattr(config, "sigreg_formulation", "lejepa_convex")
    num_slices = int(getattr(config, "sigreg_num_slices", 256))
    num_points = int(getattr(config, "sigreg_num_points", 17))
    runs = []
    for condition in args.conditions:
        source = source_for_condition[condition]
        train_features = train_data["features"][source]
        val_features = val_data["features"][source]
        for seed in args.seeds:
            positions = stratified_fraction_positions(
                train_data["cost"], args.cost_label_fraction, seed
            )
            for spec in specs:
                print(f"Training condition={condition} seed={seed} ablation={spec['name']}")
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
                    aux_label_mode=spec["label_mode"],
                    consistency_weight=spec["consistency"],
                    consistency_noise_std=args.consistency_noise_std,
                    sigreg_weight=spec["sigreg"],
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
                    "condition": condition,
                    "feature_source": source,
                    "seed": int(seed),
                    "ablation": spec["name"],
                    "aux_weight": float(spec["aux_weight"]),
                    "aux_label_mode": spec["label_mode"],
                    "consistency_weight": float(spec["consistency"]),
                    "sigreg_weight": float(spec["sigreg"]),
                    "sigreg_formulation": formulation,
                    "cost": cost,
                    "generation": generation,
                    "adapter_geometry": geometry,
                    "training_first": history[0],
                    "training_last": history[-1],
                }
                runs.append(row)
                stem = f"{condition}__{spec['name']}__seed{seed}"
                write_json(output_dir / f"{stem}.json", row)
                torch.save(
                    {
                        "state_dict": head.state_dict(),
                        "feature_mean": mean,
                        "feature_std": std,
                        "cost_mean": cost_mean,
                        "cost_std": cost_std,
                    },
                    artifact_dir / f"{stem}.pt",
                )
                del head

    summary = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_contract_hash": config.data_contract_hash,
            "vocab_hash": config.vocab_hash,
            "cost_label_fraction": args.cost_label_fraction,
            "consistency_weight": args.consistency_weight,
            "consistency_noise_std": args.consistency_noise_std,
            "sigreg_weights": args.sigreg_weights,
            "sigreg_formulation": formulation,
            "sigreg_num_slices": num_slices,
            "sigreg_num_points": num_points,
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "seeds": args.seeds,
            "conditions": args.conditions,
        },
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
