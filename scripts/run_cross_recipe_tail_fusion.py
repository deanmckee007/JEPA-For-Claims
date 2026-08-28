"""Fuse frozen representations from different pretraining recipes for tail ranking."""

import argparse
import copy
import json
import sys
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
from scripts.run_cost_attribution_ladder import collect_model_features, parse_checkpoint_spec
from scripts.run_exact_raw_cost_baselines import build_exact_history_matrix, top_frequency_columns
from scripts.run_frozen_generation_probe import resolve_device
from scripts.run_high_cost_tail_probe import (
    fit_boosted_classifier,
    fit_boosted_regressor,
    score_tail_ranking,
    top_fraction_labels,
    train_tail_threshold,
)
from scripts.run_tail_candidate_showdown import (
    aggregate_budget_runs,
    blend_candidate_scores,
    render_markdown,
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
    parser.add_argument("--training-tail-fraction", type=float, default=0.015)
    parser.add_argument(
        "--evaluation-fractions", type=float, nargs="+",
        default=[0.005, 0.01, 0.015, 0.02, 0.03, 0.05],
    )
    parser.add_argument("--rank-blend-weights", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--raw-top-columns", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args(argv)
    groups = {item["group"] for item in args.checkpoints}
    if len(groups) < 2:
        parser.error("at least two representation groups are required")
    seeds = {item["seed"] for item in args.checkpoints}
    pairs = {(item["group"], item["seed"]) for item in args.checkpoints}
    if any((group, seed) not in pairs for group in groups for seed in seeds):
        parser.error("every seed must provide every representation group")
    if args.evaluation_fractions != sorted(set(args.evaluation_fractions)):
        parser.error("evaluation fractions must be unique and increasing")
    if args.training_tail_fraction not in args.evaluation_fractions:
        parser.error("training tail fraction must be evaluated")
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


def append_budget_scores(
    runs, condition, seed, scores, val_targets, fractions, thresholds, **metadata
):
    for fraction in fractions:
        runs.append({
            "condition": condition,
            "seed": int(seed),
            "evaluation_fraction": float(fraction),
            "metrics": score_tail_ranking(
                scores, val_targets, fraction=fraction,
                train_threshold=thresholds[fraction],
            ),
            **metadata,
        })


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
    labels = top_fraction_labels(train_targets, args.training_tail_fraction)
    thresholds = {
        fraction: train_tail_threshold(train_targets, fraction)
        for fraction in args.evaluation_fractions
    }

    embeddings = {}
    dimensions = {}
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
        dimensions[group] = int(train_embedding.shape[1])

    groups = sorted({item["group"] for item in args.checkpoints})
    seeds = sorted({item["seed"] for item in args.checkpoints})
    runs = []
    for seed in seeds:
        scores_by_condition = {}
        raw_cost_scores, iterations = fit_boosted_regressor(
            raw_train, train_targets, raw_val, seed=seed, iterations=args.iterations
        )
        scores_by_condition["raw_boosted_cost"] = (raw_cost_scores, iterations)
        group_train = []
        group_val = []
        hybrid_scores = {}
        for group in groups:
            train_embedding, val_embedding = embeddings[(group, seed)]
            group_train.append(train_embedding)
            group_val.append(val_embedding)
            embedding_scores, iterations = fit_boosted_classifier(
                train_embedding, labels, val_embedding,
                seed=seed, iterations=args.iterations,
            )
            scores_by_condition[f"{group}_embedding_boosted"] = (
                embedding_scores, iterations
            )
            hybrid_train = np.concatenate([raw_train, train_embedding], axis=1)
            hybrid_val = np.concatenate([raw_val, val_embedding], axis=1)
            scores, iterations = fit_boosted_classifier(
                hybrid_train, labels, hybrid_val,
                seed=seed, iterations=args.iterations,
            )
            condition = f"raw_{group}_boosted"
            hybrid_scores[group] = scores
            scores_by_condition[condition] = (scores, iterations)

        fused_train = np.concatenate(group_train, axis=1)
        fused_val = np.concatenate(group_val, axis=1)
        fused_scores, iterations = fit_boosted_classifier(
            fused_train, labels, fused_val, seed=seed, iterations=args.iterations
        )
        scores_by_condition["all_embeddings_boosted"] = (fused_scores, iterations)
        raw_fused_scores, iterations = fit_boosted_classifier(
            np.concatenate([raw_train, fused_train], axis=1), labels,
            np.concatenate([raw_val, fused_val], axis=1),
            seed=seed, iterations=args.iterations,
        )
        scores_by_condition["raw_all_embeddings_boosted"] = (
            raw_fused_scores, iterations
        )

        if len(groups) == 2:
            first, second = groups
            for weight in args.rank_blend_weights:
                condition = f"rank_blend_{first}_{weight:g}_{second}_{1-weight:g}"
                scores_by_condition[condition] = (
                    blend_candidate_scores(
                        hybrid_scores[first], hybrid_scores[second], weight
                    ),
                    None,
                )

        for condition, (scores, iterations) in scores_by_condition.items():
            metadata = {} if iterations is None else {"iterations": iterations}
            append_budget_scores(
                runs, condition, seed, scores, val_targets,
                args.evaluation_fractions, thresholds, **metadata,
            )

    aggregate = aggregate_budget_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "mortality_label_available": False,
            "task": "high_cost_tail_proxy_cross_recipe_fusion",
            "evaluation_weights": "online",
            "training_tail_fraction": args.training_tail_fraction,
            "evaluation_fractions": args.evaluation_fractions,
            "representation_groups": groups,
            "representation_dimensions": dimensions,
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
    markdown = render_markdown(aggregate, args.evaluation_fractions)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
