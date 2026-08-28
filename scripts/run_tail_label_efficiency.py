"""Measure high-cost-tail label efficiency for raw and pretrained representations."""

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint, read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from scripts.run_cost_attribution_ladder import collect_model_features, parse_checkpoint_spec
from scripts.run_exact_raw_cost_baselines import build_exact_history_matrix, top_frequency_columns
from scripts.run_frozen_generation_probe import resolve_device
from scripts.run_high_cost_tail_probe import (
    TAIL_METRICS,
    balanced_sample_weights,
    fit_embedding_logistic,
    score_tail_ranking,
    top_fraction_labels,
    train_tail_threshold,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", dest="checkpoints", action="append",
        type=parse_checkpoint_spec, required=True,
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tail-fraction", type=float, default=0.015)
    parser.add_argument(
        "--label-fractions", type=float, nargs="+",
        default=[0.01, 0.05, 0.10, 0.25, 1.0],
    )
    parser.add_argument("--raw-top-columns", type=int, default=256)
    parser.add_argument("--boost-iterations", type=int, default=100)
    args = parser.parse_args(argv)
    if any(not 0.0 < value <= 1.0 for value in args.label_fractions):
        parser.error("label fractions must be in (0, 1]")
    groups = {item["group"] for item in args.checkpoints}
    seeds = {item["seed"] for item in args.checkpoints}
    pairs = {(item["group"], item["seed"]) for item in args.checkpoints}
    if any((group, seed) not in pairs for group in groups for seed in seeds):
        parser.error("every seed must provide every representation group")
    return args


def configure(checkpoint_path, args):
    config = copy.deepcopy(read_checkpoint_config(checkpoint_path))
    if config is None:
        raise ValueError(f"Checkpoint has no Config: {checkpoint_path}")
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.evaluation_split = "val"
    config.use_generative_save = False
    config.use_plotting = False
    config.pretrain_diffusion = False
    config.trainer_accelerator = args.accelerator
    return apply_runtime_config_overrides(config)


def stratified_binary_positions(labels, fraction, seed):
    labels = np.asarray(labels, dtype=np.int64)
    if fraction >= 1.0:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    selected = []
    for class_id in np.unique(labels):
        positions = np.flatnonzero(labels == class_id)
        count = max(1, int(math.ceil(len(positions) * fraction)))
        selected.append(rng.choice(positions, size=count, replace=False))
    return np.sort(np.concatenate(selected)).astype(np.int64)


def fit_low_label_boosted(train_x, labels, val_x, positions, *, seed, iterations):
    subset_labels = labels[positions]
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=iterations,
        max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=2.0,
        early_stopping=False, random_state=seed,
    )
    model.fit(
        train_x[positions], subset_labels,
        sample_weight=balanced_sample_weights(subset_labels),
    )
    return model.predict_proba(val_x)[:, 1]


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["condition"], run["label_fraction"])].append(run)
    output = {}
    for (condition, fraction), rows in sorted(grouped.items()):
        metrics = {}
        for metric in TAIL_METRICS:
            values = np.asarray([row["metrics"][metric] for row in rows])
            metrics[metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        output.setdefault(condition, {})[str(fraction)] = {
            "num_runs": len(rows),
            "num_labels_mean": float(np.mean([row["num_labels"] for row in rows])),
            "num_positives_mean": float(np.mean([row["num_positives"] for row in rows])),
            "metrics": metrics,
        }
    return output


def render_markdown(aggregate, fractions):
    lines = [
        "# Top-Tail Label Efficiency", "",
        "Each cell is mean precision / PR-AUC across encoder and probe seeds.", "",
        "| Condition | " + " | ".join(f"{100*f:g}% labels" for f in fractions) + " |",
        "|---|" + "---:|" * len(fractions),
    ]
    for condition, rows in aggregate.items():
        cells = []
        for fraction in fractions:
            metrics = rows[str(fraction)]["metrics"]
            cells.append(
                f"{metrics['precision_at_budget']['mean']:.3f} / "
                f"{metrics['average_precision']['mean']:.3f}"
            )
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.accelerator)
    first_config = configure(args.checkpoints[0]["path"], args)
    train_subset, _, val_subset, _, first_config, dataset = prepare_data(
        first_config, requested_eval_split="val"
    )
    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )
    raw_train, train_targets, _ = build_exact_history_matrix(
        train_subset.indices, dataset,
        future_claim_k=getattr(first_config, "future_claim_k", 0),
    )
    raw_val, val_targets, _ = build_exact_history_matrix(
        val_subset.indices, dataset,
        future_claim_k=getattr(first_config, "future_claim_k", 0),
    )
    columns = top_frequency_columns(raw_train, args.raw_top_columns)
    raw_train = raw_train[:, columns].toarray().astype(np.float32)
    raw_val = raw_val[:, columns].toarray().astype(np.float32)
    labels = top_fraction_labels(train_targets, args.tail_fraction)
    threshold = train_tail_threshold(train_targets, args.tail_fraction)

    embeddings = {}
    for checkpoint in args.checkpoints:
        group = checkpoint["group"]
        seed = checkpoint["seed"]
        config = configure(checkpoint["path"], args)
        pl.seed_everything(seed, workers=True)
        model = load_claims_model_checkpoint(
            HierarchicalClaimsModel, checkpoint["path"], config=config,
            map_location=device,
        )
        train_embedding, model_train_targets = collect_model_features(
            model, train_loader, source="patient_representation_pre_sae", device=device
        )
        val_embedding, model_val_targets = collect_model_features(
            model, val_loader, source="patient_representation_pre_sae", device=device
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if not np.allclose(model_train_targets, train_targets) or not np.allclose(
            model_val_targets, val_targets
        ):
            raise ValueError("Raw and representation target order does not match")
        embeddings[(group, seed)] = (train_embedding, val_embedding)

    runs = []
    groups = sorted({item["group"] for item in args.checkpoints})
    seeds = sorted({item["seed"] for item in args.checkpoints})
    for seed in seeds:
        feature_sets = {"raw": (raw_train, raw_val)}
        for group in groups:
            train_embedding, val_embedding = embeddings[(group, seed)]
            feature_sets[group] = (train_embedding, val_embedding)
            feature_sets[f"raw_{group}"] = (
                np.concatenate([raw_train, train_embedding], axis=1),
                np.concatenate([raw_val, val_embedding], axis=1),
            )
        for fraction in sorted(set(args.label_fractions)):
            positions = stratified_binary_positions(labels, fraction, seed)
            subset_labels = labels[positions]
            for feature_name, (train_x, val_x) in feature_sets.items():
                logistic_scores = fit_embedding_logistic(
                    train_x[positions], subset_labels, val_x, seed=seed
                )
                boosted_scores = fit_low_label_boosted(
                    train_x, labels, val_x, positions,
                    seed=seed, iterations=args.boost_iterations,
                )
                for learner, scores in (
                    ("logistic", logistic_scores), ("boosted", boosted_scores)
                ):
                    runs.append({
                        "condition": f"{feature_name}_{learner}",
                        "seed": int(seed),
                        "label_fraction": float(fraction),
                        "num_labels": int(len(positions)),
                        "num_positives": int(subset_labels.sum()),
                        "metrics": score_tail_ranking(
                            scores, val_targets, fraction=args.tail_fraction,
                            train_threshold=threshold,
                        ),
                    })

    fractions = sorted(set(args.label_fractions))
    aggregate = aggregate_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "mortality_label_available": False,
            "task": "high_cost_tail_proxy_label_efficiency",
            "tail_fraction": args.tail_fraction,
            "label_fractions": fractions,
            "stratified_subsampling": True,
            "evaluation_weights": "online",
            "raw_feature_count": int(len(columns)),
            "data_contract_hash": first_config.data_contract_hash,
            "vocab_hash": first_config.vocab_hash,
        },
        "aggregate": aggregate,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(aggregate, fractions)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
