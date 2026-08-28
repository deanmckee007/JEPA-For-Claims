"""Evaluate high-cost member identification at a fixed top-tail operating budget."""

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
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, ndcg_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint, read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from scripts.run_cost_attribution_ladder import (
    collect_model_features,
    parse_checkpoint_spec,
)
from scripts.run_exact_raw_cost_baselines import (
    build_exact_history_matrix,
    top_frequency_columns,
)
from scripts.run_frozen_generation_probe import resolve_device


TAIL_METRICS = (
    "precision_at_budget",
    "recall_at_budget",
    "lift_at_budget",
    "average_precision",
    "binary_ndcg_at_budget",
    "cost_capture_at_budget",
    "cost_lift_at_budget",
    "threshold_precision_at_budget",
    "threshold_recall_at_budget",
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
        "--ordinal-fractions", type=float, nargs="+",
        default=[0.005, 0.015, 0.05, 0.20],
        help=(
            "Nested upper-tail boundaries for ordinal cost classes, ordered "
            "from most to least extreme."
        ),
    )
    parser.add_argument("--raw-top-columns", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args(argv)
    if not 0.0 < args.tail_fraction < 0.5:
        parser.error("--tail-fraction must be in (0, 0.5)")
    if (
        any(not 0.0 < value < 0.5 for value in args.ordinal_fractions)
        or args.ordinal_fractions != sorted(set(args.ordinal_fractions))
    ):
        parser.error("--ordinal-fractions must be unique, increasing values in (0, 0.5)")
    if args.tail_fraction not in args.ordinal_fractions:
        parser.error("--tail-fraction must be one of --ordinal-fractions")
    if args.raw_top_columns <= 0 or args.iterations <= 0:
        parser.error("feature count and iterations must be positive")
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


def top_fraction_labels(targets, fraction):
    targets = np.asarray(targets)
    count = max(1, int(math.ceil(len(targets) * fraction)))
    order = np.argsort(targets, kind="mergesort")
    labels = np.zeros(len(targets), dtype=np.int64)
    labels[order[-count:]] = 1
    return labels


def train_tail_threshold(targets, fraction):
    labels = top_fraction_labels(targets, fraction)
    return float(np.min(np.asarray(targets)[labels == 1]))


def ordinal_cost_labels(targets, fractions):
    """Assign ordered upper-tail bands, with zero denoting the non-tail class."""
    targets = np.asarray(targets)
    labels = np.zeros(len(targets), dtype=np.int64)
    order = np.argsort(targets, kind="mergesort")
    fractions = sorted(fractions, reverse=True)
    for level, fraction in enumerate(fractions, start=1):
        count = max(1, int(math.ceil(len(targets) * fraction)))
        labels[order[-count:]] = level
    return labels


def ordinal_class_thresholds(targets, fractions):
    targets = np.asarray(targets)
    return {
        str(fraction): train_tail_threshold(targets, fraction)
        for fraction in fractions
    }


def score_tail_ranking(scores, targets, *, fraction, train_threshold):
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    labels = top_fraction_labels(targets, fraction)
    threshold_labels = (targets >= train_threshold).astype(np.int64)
    count = int(labels.sum())
    selected = np.argsort(scores, kind="mergesort")[-count:]
    selected_labels = labels[selected]
    precision = float(selected_labels.mean())
    recall = float(selected_labels.sum() / max(labels.sum(), 1))
    prevalence = float(labels.mean())
    target_dollars = np.expm1(targets)
    cost_capture = float(target_dollars[selected].sum() / target_dollars.sum())
    threshold_selected = threshold_labels[selected]
    return {
        "selection_count": count,
        "validation_samples": int(len(targets)),
        "precision_at_budget": precision,
        "recall_at_budget": recall,
        "lift_at_budget": float(precision / prevalence),
        "average_precision": float(average_precision_score(labels, scores)),
        "binary_ndcg_at_budget": float(ndcg_score(labels[None, :], scores[None, :], k=count)),
        "cost_capture_at_budget": cost_capture,
        "cost_lift_at_budget": float(cost_capture / (count / len(targets))),
        "threshold_positive_prevalence": float(threshold_labels.mean()),
        "threshold_precision_at_budget": float(threshold_selected.mean()),
        "threshold_recall_at_budget": float(
            threshold_selected.sum() / max(threshold_labels.sum(), 1)
        ),
    }


def balanced_sample_weights(labels):
    labels = np.asarray(labels)
    weights = np.ones(len(labels), dtype=np.float32)
    classes, counts = np.unique(labels, return_counts=True)
    largest_count = int(counts.max())
    for class_id, count in zip(classes, counts):
        weights[labels == class_id] = largest_count / max(int(count), 1)
    return weights


def fit_boosted_regressor(train_x, train_y, val_x, *, seed, iterations):
    model = HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.05, max_iter=iterations,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0,
        early_stopping=True, validation_fraction=0.15, random_state=seed,
    )
    model.fit(train_x, train_y)
    return model.predict(val_x), int(model.n_iter_)


def fit_boosted_classifier(train_x, train_labels, val_x, *, seed, iterations):
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=iterations,
        max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=2.0,
        early_stopping=True, validation_fraction=0.15, random_state=seed,
    )
    model.fit(train_x, train_labels, sample_weight=balanced_sample_weights(train_labels))
    return model.predict_proba(val_x)[:, 1], int(model.n_iter_)


def fit_embedding_logistic(train_x, train_labels, val_x, *, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            solver="liblinear", random_state=seed,
        ),
    )
    model.fit(train_x, train_labels)
    return model.predict_proba(val_x)[:, 1]


def fit_ordinal_boosted_classifier(train_x, train_labels, val_x, *, seed, iterations):
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=iterations,
        max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=2.0,
        early_stopping=True, validation_fraction=0.15, random_state=seed,
    )
    model.fit(train_x, train_labels, sample_weight=balanced_sample_weights(train_labels))
    return model.predict_proba(val_x), model.classes_, int(model.n_iter_)


def fit_ordinal_embedding_logistic(train_x, train_labels, val_x, *, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            solver="lbfgs", random_state=seed,
        ),
    )
    model.fit(train_x, train_labels)
    classifier = model.named_steps["logisticregression"]
    return model.predict_proba(val_x), classifier.classes_


def ordinal_ranking_scores(probabilities, classes, *, tail_class_minimum):
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes)
    severity = probabilities @ classes.astype(np.float64)
    tail_probability = probabilities[:, classes >= tail_class_minimum].sum(axis=1)
    return {"severity": severity, "tail_probability": tail_probability}


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["condition"]].append(run)
    output = {}
    for condition, rows in sorted(grouped.items()):
        metrics = {}
        for metric in TAIL_METRICS:
            values = np.asarray([row["metrics"][metric] for row in rows], dtype=np.float64)
            metrics[metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        output[condition] = {"num_runs": len(rows), "metrics": metrics}
    return output


def render_markdown(aggregate, *, tail_fraction=0.015):
    percentage = 100.0 * tail_fraction
    lines = [
        f"# High-Cost Tail Probe (Top {percentage:g}%)", "",
        "| Condition | Runs | Precision | Recall | Lift | PR-AUC | NDCG@budget | Cost capture | Cost lift |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, row in aggregate.items():
        m = row["metrics"]
        mean = lambda name: m[name]["mean"]
        lines.append(
            f"| {condition} | {row['num_runs']} | {mean('precision_at_budget'):.4f} | "
            f"{mean('recall_at_budget'):.4f} | {mean('lift_at_budget'):.2f} | "
            f"{mean('average_precision'):.4f} | {mean('binary_ndcg_at_budget'):.4f} | "
            f"{mean('cost_capture_at_budget'):.4f} | {mean('cost_lift_at_budget'):.2f} |"
        )
    return "\n".join(lines) + "\n"


def append_run(runs, condition, seed, scores, val_targets, fraction, threshold, **metadata):
    runs.append(
        {
            "condition": condition,
            "seed": int(seed),
            "metrics": score_tail_ranking(
                scores, val_targets, fraction=fraction, train_threshold=threshold
            ),
            **metadata,
        }
    )


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
    train_labels = top_fraction_labels(train_targets, args.tail_fraction)
    threshold = train_tail_threshold(train_targets, args.tail_fraction)
    ordinal_labels = ordinal_cost_labels(train_targets, args.ordinal_fractions)
    tail_class_minimum = (
        len(args.ordinal_fractions) - args.ordinal_fractions.index(args.tail_fraction)
    )

    runs = []
    embeddings = {}
    for checkpoint in args.checkpoints:
        seed = checkpoint["seed"]
        group = checkpoint["group"]
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
            raise ValueError("Raw and embedding target order does not match")
        embeddings[(group, seed)] = (train_embedding, val_embedding)

    seeds = sorted({item["seed"] for item in args.checkpoints})
    groups = sorted({item["group"] for item in args.checkpoints})
    for seed in seeds:
        raw_regression_scores, iterations = fit_boosted_regressor(
            raw_train, train_targets, raw_val, seed=seed, iterations=args.iterations
        )
        append_run(
            runs, "raw_boosted_cost", seed, raw_regression_scores, val_targets,
            args.tail_fraction, threshold, iterations=iterations,
        )
        raw_tail_scores, iterations = fit_boosted_classifier(
            raw_train, train_labels, raw_val, seed=seed, iterations=args.iterations
        )
        append_run(
            runs, "raw_boosted_tail", seed, raw_tail_scores, val_targets,
            args.tail_fraction, threshold, iterations=iterations,
        )
        raw_ordinal_probabilities, ordinal_classes, iterations = (
            fit_ordinal_boosted_classifier(
                raw_train, ordinal_labels, raw_val, seed=seed,
                iterations=args.iterations,
            )
        )
        for score_name, scores in ordinal_ranking_scores(
            raw_ordinal_probabilities, ordinal_classes,
            tail_class_minimum=tail_class_minimum,
        ).items():
            append_run(
                runs, f"raw_boosted_ordinal_{score_name}", seed, scores,
                val_targets, args.tail_fraction, threshold, iterations=iterations,
            )
        for group in groups:
            key = (group, seed)
            if key not in embeddings:
                continue
            train_embedding, val_embedding = embeddings[key]
            logistic_scores = fit_embedding_logistic(
                train_embedding, train_labels, val_embedding, seed=seed
            )
            append_run(
                runs, f"{group}_embedding_tail", seed, logistic_scores, val_targets,
                args.tail_fraction, threshold,
            )
            ordinal_probabilities, ordinal_classes = fit_ordinal_embedding_logistic(
                train_embedding, ordinal_labels, val_embedding, seed=seed
            )
            for score_name, scores in ordinal_ranking_scores(
                ordinal_probabilities, ordinal_classes,
                tail_class_minimum=tail_class_minimum,
            ).items():
                append_run(
                    runs, f"{group}_embedding_ordinal_{score_name}", seed,
                    scores, val_targets, args.tail_fraction, threshold,
                )
            hybrid_train = np.concatenate([raw_train, train_embedding], axis=1)
            hybrid_val = np.concatenate([raw_val, val_embedding], axis=1)
            hybrid_tail_scores, iterations = fit_boosted_classifier(
                hybrid_train, train_labels, hybrid_val, seed=seed,
                iterations=args.iterations,
            )
            append_run(
                runs, f"{group}_hybrid_tail", seed, hybrid_tail_scores, val_targets,
                args.tail_fraction, threshold, iterations=iterations,
            )
            hybrid_ordinal_probabilities, ordinal_classes, iterations = (
                fit_ordinal_boosted_classifier(
                    hybrid_train, ordinal_labels, hybrid_val, seed=seed,
                    iterations=args.iterations,
                )
            )
            for score_name, scores in ordinal_ranking_scores(
                hybrid_ordinal_probabilities, ordinal_classes,
                tail_class_minimum=tail_class_minimum,
            ).items():
                append_run(
                    runs, f"{group}_hybrid_ordinal_{score_name}", seed,
                    scores, val_targets, args.tail_fraction, threshold,
                    iterations=iterations,
                )
            hybrid_cost_scores, iterations = fit_boosted_regressor(
                hybrid_train, train_targets, hybrid_val, seed=seed,
                iterations=args.iterations,
            )
            append_run(
                runs, f"{group}_hybrid_cost", seed, hybrid_cost_scores, val_targets,
                args.tail_fraction, threshold, iterations=iterations,
            )

    aggregate = aggregate_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "mortality_label_available": False,
            "task": "high_cost_tail_proxy",
            "tail_fraction": args.tail_fraction,
            "selection_count": int(math.ceil(len(val_targets) * args.tail_fraction)),
            "train_tail_count": int(train_labels.sum()),
            "train_tail_threshold_log1p_cost": threshold,
            "train_tail_threshold_dollars": float(np.expm1(threshold)),
            "ordinal_fractions": args.ordinal_fractions,
            "ordinal_class_counts": {
                str(int(class_id)): int((ordinal_labels == class_id).sum())
                for class_id in np.unique(ordinal_labels)
            },
            "ordinal_thresholds_log1p_cost": ordinal_class_thresholds(
                train_targets, args.ordinal_fractions
            ),
            "ordinal_tail_class_minimum": int(tail_class_minimum),
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
    markdown = render_markdown(aggregate, tail_fraction=args.tail_fraction)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
