"""Compare and blend the two surviving high-cost tail ranking candidates."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import rankdata
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
    fit_boosted_classifier,
    fit_boosted_regressor,
    score_tail_ranking,
    top_fraction_labels,
    train_tail_threshold,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", dest="checkpoints", action="append",
        type=parse_checkpoint_spec, required=True,
        help="Dense checkpoint as dense_seedN=PATH. Repeat for each seed.",
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--training-tail-fraction", type=float, default=0.015)
    parser.add_argument(
        "--evaluation-fractions", type=float, nargs="+",
        default=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
    )
    parser.add_argument(
        "--raw-blend-weights", type=float, nargs="+", default=[0.25, 0.50, 0.75],
        help="Predeclared raw-cost rank weights; the remainder is hybrid-tail rank.",
    )
    parser.add_argument("--raw-top-columns", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args(argv)
    if any(not 0.0 < value < 0.5 for value in args.evaluation_fractions):
        parser.error("evaluation fractions must be in (0, 0.5)")
    if args.evaluation_fractions != sorted(set(args.evaluation_fractions)):
        parser.error("evaluation fractions must be unique and increasing")
    if args.training_tail_fraction not in args.evaluation_fractions:
        parser.error("training tail fraction must be one of the evaluation fractions")
    if any(not 0.0 < value < 1.0 for value in args.raw_blend_weights):
        parser.error("blend weights must be in (0, 1)")
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


def percentile_ranks(scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size <= 1:
        return np.zeros_like(scores)
    return (rankdata(scores, method="average") - 1.0) / (scores.size - 1.0)


def blend_candidate_scores(raw_cost_scores, hybrid_tail_scores, raw_weight):
    return (
        raw_weight * percentile_ranks(raw_cost_scores)
        + (1.0 - raw_weight) * percentile_ranks(hybrid_tail_scores)
    )


def selected_indices(scores, count):
    return np.argsort(np.asarray(scores), kind="mergesort")[-count:]


def aggregate_budget_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["condition"], run["evaluation_fraction"])].append(run)
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
            "num_runs": len(rows), "metrics": metrics,
        }
    return output


def render_markdown(aggregate, fractions):
    run_counts = {
        row["num_runs"] for condition in aggregate.values() for row in condition.values()
    }
    run_label = (
        f"{next(iter(run_counts))} run(s) per condition"
        if len(run_counts) == 1
        else "the available runs"
    )
    lines = [
        "# Tail Candidate Budget Showdown", "",
        f"Each cell is mean precision / future-cost capture across {run_label}.", "",
        "| Condition | " + " | ".join(f"Top {100*f:g}%" for f in fractions) + " |",
        "|---|" + "---:|" * len(fractions),
    ]
    for condition, rows in aggregate.items():
        cells = []
        for fraction in fractions:
            metrics = rows[str(fraction)]["metrics"]
            precision = metrics["precision_at_budget"]["mean"]
            capture = metrics["cost_capture_at_budget"]["mean"]
            cells.append(f"{precision:.3f} / {capture:.3f}")
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
    train_labels = top_fraction_labels(train_targets, args.training_tail_fraction)
    thresholds = {
        fraction: train_tail_threshold(train_targets, fraction)
        for fraction in args.evaluation_fractions
    }

    embeddings = {}
    for checkpoint in args.checkpoints:
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
            raise ValueError("Raw and embedding target order does not match")
        embeddings[seed] = (train_embedding, val_embedding)

    runs = []
    overlap = []
    for seed in sorted(embeddings):
        train_embedding, val_embedding = embeddings[seed]
        raw_cost_scores, raw_iterations = fit_boosted_regressor(
            raw_train, train_targets, raw_val, seed=seed, iterations=args.iterations
        )
        hybrid_train = np.concatenate([raw_train, train_embedding], axis=1)
        hybrid_val = np.concatenate([raw_val, val_embedding], axis=1)
        hybrid_tail_scores, hybrid_iterations = fit_boosted_classifier(
            hybrid_train, train_labels, hybrid_val, seed=seed,
            iterations=args.iterations,
        )
        candidates = {
            "raw_boosted_cost": raw_cost_scores,
            "dense_hybrid_tail": hybrid_tail_scores,
        }
        for weight in args.raw_blend_weights:
            candidates[f"rank_blend_raw_{weight:g}"] = blend_candidate_scores(
                raw_cost_scores, hybrid_tail_scores, weight
            )
        for condition, scores in candidates.items():
            for fraction in args.evaluation_fractions:
                runs.append({
                    "condition": condition,
                    "seed": int(seed),
                    "evaluation_fraction": float(fraction),
                    "metrics": score_tail_ranking(
                        scores, val_targets, fraction=fraction,
                        train_threshold=thresholds[fraction],
                    ),
                })
        count = int(np.ceil(len(val_targets) * args.training_tail_fraction))
        raw_selected = set(selected_indices(raw_cost_scores, count).tolist())
        hybrid_selected = set(selected_indices(hybrid_tail_scores, count).tolist())
        truth = set(np.flatnonzero(top_fraction_labels(
            val_targets, args.training_tail_fraction
        )).tolist())
        overlap.append({
            "seed": int(seed),
            "selection_count": count,
            "raw_hybrid_selection_overlap": len(raw_selected & hybrid_selected),
            "raw_only_true_positives": len((raw_selected - hybrid_selected) & truth),
            "hybrid_only_true_positives": len((hybrid_selected - raw_selected) & truth),
            "raw_iterations": raw_iterations,
            "hybrid_iterations": hybrid_iterations,
        })

    aggregate = aggregate_budget_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "mortality_label_available": False,
            "task": "high_cost_tail_proxy_candidate_showdown",
            "training_tail_fraction": args.training_tail_fraction,
            "evaluation_fractions": args.evaluation_fractions,
            "raw_blend_weights": args.raw_blend_weights,
            "blend_weights_predeclared": True,
            "raw_feature_count": int(len(columns)),
            "data_contract_hash": first_config.data_contract_hash,
            "vocab_hash": first_config.vocab_hash,
        },
        "aggregate": aggregate,
        "selection_overlap_at_training_budget": overlap,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(aggregate, args.evaluation_fractions)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
