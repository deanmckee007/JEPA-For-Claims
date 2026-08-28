"""Compare semantic generation supervision with label-free consistency regularization."""

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
    parser.add_argument("--consistency-weights", nargs="+", type=float, default=[1.0, 10.0, 100.0])
    parser.add_argument("--consistency-noise-std", type=float, default=0.1)
    parser.add_argument("--skip-reference-arms", action="store_true")
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--conditions", nargs="+", choices=["pretrained", "raw_history"],
        default=["pretrained", "raw_history"],
    )
    return parser.parse_args(argv)


def build_specs(args):
    reference_specs = [
        {"name": "cost_only", "aux_weight": 0.0, "label_mode": "aligned", "consistency_weight": 0.0},
        {"name": "aligned_all", "aux_weight": args.aux_weight, "label_mode": "aligned", "consistency_weight": 0.0},
        {"name": "shuffled_all", "aux_weight": args.aux_weight, "label_mode": "shuffled", "consistency_weight": 0.0},
    ]
    if args.reference_only:
        return reference_specs
    specs = [] if args.skip_reference_arms else reference_specs
    specs.extend(
        {
            "name": f"consistency_{weight:g}",
            "aux_weight": 0.0,
            "label_mode": "aligned",
            "consistency_weight": float(weight),
        }
        for weight in args.consistency_weights
    )
    return specs


def aggregate(runs):
    grouped = defaultdict(list)
    for row in runs:
        grouped[(row["condition"], row["ablation"])].append(row)
    result = {}
    for (condition, ablation), rows in grouped.items():
        values = {}
        for metric in ("target_probe_mae_dollars", "target_probe_rmse_dollars", "target_probe_wape_percent"):
            observed = [row["cost"][metric] for row in rows]
            values[metric] = {
                "mean": float(np.mean(observed)),
                "sample_std": float(np.std(observed, ddof=1)) if len(observed) > 1 else 0.0,
            }
        result[f"{condition}|{ablation}"] = {"num_runs": len(rows), "cost": values}
    return result


def paired_deltas(runs):
    by_key = {(row["condition"], row["ablation"], row["seed"]): row for row in runs}
    result = {}
    for condition in sorted({row["condition"] for row in runs}):
        ablations = sorted({row["ablation"] for row in runs if row["ablation"] != "cost_only"})
        for ablation in ablations:
            deltas = []
            for seed in sorted({row["seed"] for row in runs}):
                baseline = by_key.get((condition, "cost_only", seed))
                candidate = by_key.get((condition, ablation, seed))
                if baseline and candidate:
                    deltas.append(
                        candidate["cost"]["target_probe_mae_dollars"]
                        - baseline["cost"]["target_probe_mae_dollars"]
                    )
            if deltas:
                result[f"{condition}|{ablation}"] = {
                    "mae_delta_mean": float(np.mean(deltas)),
                    "sample_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "wins": int(sum(delta < 0 for delta in deltas)),
                    "paired_seeds": len(deltas),
                }
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
                print(
                    f"Training condition={condition} seed={seed} "
                    f"ablation={spec['name']}"
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
                    aux_label_mode=spec["label_mode"],
                    consistency_weight=spec["consistency_weight"],
                    consistency_noise_std=args.consistency_noise_std,
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
                    "consistency_weight": float(spec["consistency_weight"]),
                    "consistency_noise_std": float(args.consistency_noise_std),
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
            "data_contract": str(Path(args.data_contract).resolve()),
            "data_contract_hash": config.data_contract_hash,
            "vocab_hash": config.vocab_hash,
            "cost_label_fraction": args.cost_label_fraction,
            "aux_weight": args.aux_weight,
            "consistency_weights": args.consistency_weights,
            "consistency_noise_std": args.consistency_noise_std,
            "skip_reference_arms": args.skip_reference_arms,
            "reference_only": args.reference_only,
            "consistency_uses_full_training_features": True,
            "raw_history_target_excluded": True,
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "seeds": args.seeds,
            "conditions": args.conditions,
        },
        "runs": runs,
        "aggregate": aggregate(runs),
        "paired_cost_deltas": paired_deltas(runs),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(
        {"aggregate": summary["aggregate"], "paired_cost_deltas": summary["paired_cost_deltas"]},
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
