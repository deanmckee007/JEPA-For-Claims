"""Attribute frozen cost-probe value to raw history, architecture, or SSL pretraining."""

import argparse
import copy
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint, read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import (
    compute_heldout_regression_probe_metrics,
    score_regression_predictions,
    select_representation_tensor,
)
from scripts.run_cost_supervision_ablation import stratified_fraction_positions
from scripts.run_frozen_generation_probe import hashed_raw_history_features, resolve_device


METRICS = (
    "target_probe_mae_dollars",
    "target_probe_rmse_dollars",
    "target_probe_rmse_log1p",
    "target_probe_wape_percent",
)


def parse_checkpoint_spec(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    match = re.fullmatch(r"(.+)_seed(\d+)", name)
    if match is None:
        raise argparse.ArgumentTypeError(
            "checkpoint NAME must end in _seedN (for example dense_seed42)"
        )
    return {
        "name": name,
        "group": match.group(1),
        "seed": int(match.group(2)),
        "path": path,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        type=parse_checkpoint_spec,
        required=True,
        help="Named checkpoint as GROUP_seedN=PATH. Repeat for each checkpoint.",
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--raw-hash-dim", type=int, default=128)
    parser.add_argument(
        "--representation-source",
        choices=["patient_representation_pre_sae", "next_claim_prediction"],
        default="patient_representation_pre_sae",
    )
    parser.add_argument(
        "--label-fractions", nargs="+", type=float,
        default=[0.01, 0.05, 0.10, 0.25, 1.0],
    )
    parser.add_argument(
        "--include-shuffled", action="store_true",
        help="Fit a negative control after permuting train features away from cost labels.",
    )
    parser.add_argument(
        "--skip-random-encoder", action="store_true",
        help="Skip matched randomly initialized encoders.",
    )
    parser.add_argument("--random-seed-offset", type=int, default=314159)
    args = parser.parse_args(argv)
    if any(not 0.0 < value <= 1.0 for value in args.label_fractions):
        parser.error("--label-fractions must be in (0, 1]")
    if args.raw_hash_dim < 8:
        parser.error("--raw-hash-dim must be at least 8")
    names = [item["name"] for item in args.checkpoints]
    if len(names) != len(set(names)):
        parser.error("checkpoint names must be unique")
    return args


def configure_checkpoint(checkpoint_path, args):
    config = copy.deepcopy(read_checkpoint_config(checkpoint_path))
    if config is None:
        raise ValueError(f"Checkpoint has no saved Config: {checkpoint_path}")
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.evaluation_split = "val"
    config.use_generative_save = False
    config.use_plotting = False
    config.pretrain_diffusion = False
    config.trainer_accelerator = args.accelerator
    return apply_runtime_config_overrides(config)


def make_eval_loader(subset, dataset, batch_size):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )


def collect_raw_features(loader, *, future_claim_k, output_dim):
    hash_chunks = []
    target_chunks = []
    for cpt, icd, ttnc, target in loader:
        hash_chunks.append(
            hashed_raw_history_features(
                cpt,
                icd,
                ttnc,
                future_claim_k=future_claim_k,
                output_dim=output_dim,
            )
        )
        target_chunks.append(target.numpy())
    hashed = np.concatenate(hash_chunks).astype(np.float32)
    return {
        "history_statistics": hashed[:, :4].copy(),
        "raw_history_hash": hashed,
        "cost": np.concatenate(target_chunks).astype(np.float32),
    }


def collect_model_features(model, loader, *, source, device):
    model = model.to(device)
    model.eval()
    feature_chunks = []
    target_chunks = []
    with torch.no_grad():
        for cpt, icd, ttnc, target in loader:
            outputs = model(
                cpt_tensor=cpt.to(device),
                icd_tensor=icd.to(device),
                ttnc_tensor=ttnc.to(device),
                target=target.to(device),
                teacher_forcing=True,
                generation=False,
            )
            feature_chunks.append(
                select_representation_tensor(outputs, source).detach().cpu().numpy()
            )
            target_chunks.append(target.numpy())
    return (
        np.concatenate(feature_chunks).astype(np.float32),
        np.concatenate(target_chunks).astype(np.float32),
    )


def constant_metrics(train_targets, eval_targets, positions):
    prediction = np.full(
        len(eval_targets),
        float(np.mean(train_targets[positions])),
        dtype=np.float32,
    )
    return score_regression_predictions(prediction, eval_targets)


def probe_metrics(train_features, train_targets, eval_features, eval_targets, positions):
    return compute_heldout_regression_probe_metrics(
        train_features[positions],
        train_targets[positions],
        eval_features,
        eval_targets,
    )


def make_run(condition, fraction, seed, num_labels, metrics, **metadata):
    return {
        "condition": condition,
        "label_fraction": float(fraction),
        "seed": int(seed),
        "num_labels": int(num_labels),
        "metrics": metrics,
        **metadata,
    }


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["condition"], run["label_fraction"])].append(run)
    aggregate = {}
    for (condition, fraction), rows in sorted(grouped.items()):
        metrics = {}
        for metric in METRICS:
            values = [float(row["metrics"][metric]) for row in rows]
            metrics[metric] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        aggregate[f"{condition}|{fraction:g}"] = {
            "num_runs": len(rows),
            "metrics": metrics,
        }
    return aggregate


def render_markdown(aggregate):
    lines = [
        "# Cost Attribution Ladder",
        "",
        "| Condition | Labels | Runs | MAE ($) | RMSE ($) | Log RMSE | WAPE (%) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in aggregate.items():
        condition, fraction = key.rsplit("|", 1)
        metric = row["metrics"]
        lines.append(
            f"| {condition} | {100 * float(fraction):g}% | {row['num_runs']} | "
            f"{metric['target_probe_mae_dollars']['mean']:.2f} | "
            f"{metric['target_probe_rmse_dollars']['mean']:.2f} | "
            f"{metric['target_probe_rmse_log1p']['mean']:.4f} | "
            f"{metric['target_probe_wape_percent']['mean']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_probe_runs(
    runs,
    *,
    condition,
    train_features,
    val_features,
    train_targets,
    val_targets,
    fractions,
    seed,
    metadata=None,
    shuffle=False,
):
    metadata = metadata or {}
    if shuffle:
        permutation = np.random.default_rng(seed + 9017).permutation(len(train_features))
        train_features = train_features[permutation]
    for fraction in fractions:
        positions = stratified_fraction_positions(train_targets, fraction, seed)
        metrics = probe_metrics(
            train_features,
            train_targets,
            val_features,
            val_targets,
            positions,
        )
        runs.append(
            make_run(condition, fraction, seed, len(positions), metrics, **metadata)
        )


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.accelerator)
    fractions = sorted(set(args.label_fractions))

    first_config = configure_checkpoint(args.checkpoints[0]["path"], args)
    train_subset, _, val_subset, _, first_config, dataset = prepare_data(
        first_config,
        requested_eval_split="val",
    )
    train_loader = make_eval_loader(train_subset, dataset, args.batch_size)
    val_loader = make_eval_loader(val_subset, dataset, args.batch_size)
    raw_train = collect_raw_features(
        train_loader,
        future_claim_k=getattr(first_config, "future_claim_k", 0),
        output_dim=args.raw_hash_dim,
    )
    raw_val = collect_raw_features(
        val_loader,
        future_claim_k=getattr(first_config, "future_claim_k", 0),
        output_dim=args.raw_hash_dim,
    )

    run_seeds = sorted({item["seed"] for item in args.checkpoints})
    runs = []
    for seed in run_seeds:
        for fraction in fractions:
            positions = stratified_fraction_positions(raw_train["cost"], fraction, seed)
            runs.append(
                make_run(
                    "constant_log_mean",
                    fraction,
                    seed,
                    len(positions),
                    constant_metrics(raw_train["cost"], raw_val["cost"], positions),
                )
            )
        for source in ("history_statistics", "raw_history_hash"):
            append_probe_runs(
                runs,
                condition=source,
                train_features=raw_train[source],
                val_features=raw_val[source],
                train_targets=raw_train["cost"],
                val_targets=raw_val["cost"],
                fractions=fractions,
                seed=seed,
            )

    for checkpoint in args.checkpoints:
        seed = checkpoint["seed"]
        group = checkpoint["group"]
        config = configure_checkpoint(checkpoint["path"], args)
        pl.seed_everything(seed, workers=True)
        model = load_claims_model_checkpoint(
            HierarchicalClaimsModel,
            checkpoint["path"],
            config=config,
            map_location=device,
        )
        train_features, train_targets = collect_model_features(
            model, train_loader, source=args.representation_source, device=device
        )
        val_features, val_targets = collect_model_features(
            model, val_loader, source=args.representation_source, device=device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        metadata = {
            "checkpoint": str(Path(checkpoint["path"]).resolve()),
            "representation_source": args.representation_source,
        }
        append_probe_runs(
            runs,
            condition=f"{group}_pretrained",
            train_features=train_features,
            val_features=val_features,
            train_targets=train_targets,
            val_targets=val_targets,
            fractions=fractions,
            seed=seed,
            metadata=metadata,
        )
        if args.include_shuffled:
            append_probe_runs(
                runs,
                condition=f"{group}_shuffled",
                train_features=train_features,
                val_features=val_features,
                train_targets=train_targets,
                val_targets=val_targets,
                fractions=fractions,
                seed=seed,
                metadata=metadata,
                shuffle=True,
            )
        if not args.skip_random_encoder:
            random_seed = args.random_seed_offset + seed
            pl.seed_everything(random_seed, workers=True)
            random_model = HierarchicalClaimsModel(copy.deepcopy(config))
            random_train_features, random_train_targets = collect_model_features(
                random_model, train_loader, source=args.representation_source, device=device
            )
            random_val_features, random_val_targets = collect_model_features(
                random_model, val_loader, source=args.representation_source, device=device
            )
            del random_model
            if device.type == "cuda":
                torch.cuda.empty_cache()
            append_probe_runs(
                runs,
                condition=f"{group}_random_encoder",
                train_features=random_train_features,
                val_features=random_val_features,
                train_targets=random_train_targets,
                val_targets=random_val_targets,
                fractions=fractions,
                seed=seed,
                metadata={
                    "random_encoder_seed": random_seed,
                    "representation_source": args.representation_source,
                },
            )

    aggregate = aggregate_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "data_contract": str(Path(args.data_contract).resolve()),
            "data_contract_hash": first_config.data_contract_hash,
            "vocab_hash": first_config.vocab_hash,
            "train_samples": int(len(raw_train["cost"])),
            "validation_samples": int(len(raw_val["cost"])),
            "representation_source": args.representation_source,
            "probe": "standardized_ridge_alpha_1_on_log1p_cost",
            "raw_history_excludes_held_out_next_claim": True,
            "label_fractions": fractions,
            "run_seeds": run_seeds,
        },
        "aggregate": aggregate,
        "runs": runs,
    }
    write_json(output_dir / "summary.json", payload)
    (output_dir / "summary.md").write_text(
        render_markdown(aggregate), encoding="utf-8"
    )
    print(render_markdown(aggregate))


if __name__ == "__main__":
    main()
