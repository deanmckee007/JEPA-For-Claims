import argparse
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
from jepa_utils.config import (
    Config,
    apply_config_overrides,
    apply_runtime_config_overrides,
    apply_training_recipe,
    get_training_recipe_names,
)
from jepa_utils.data_prep import prepare_data
from jepa_utils.checkpointing import load_claims_model_checkpoint
from jepa_utils.representation_eval import (
    collect_patient_representations,
    compute_heldout_regression_probe_metrics,
    get_representation_source_names,
    score_regression_predictions,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate JEPA checkpoints with cost-only downstream probes."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to load.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to the parquet dataset.")
    parser.add_argument("--data-contract", type=str, default=None, help="Frozen split/vocabulary contract.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument(
        "--recipe",
        choices=get_training_recipe_names(),
        default="custom",
        help="Recipe preset used to reconstruct the model config.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device selection for embedding extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding extraction.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to score.")
    parser.add_argument(
        "--representation-source",
        choices=get_representation_source_names(),
        default="patient_representation_pre_sae",
        help="Which representation to feed into the cost probe.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for evaluation and probes.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="cost_probe.json",
        help="Path to write the evaluation report.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        help="Config override in key=value form. Repeat to set multiple fields.",
    )
    return parser.parse_args(argv)


def resolve_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested for evaluation, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_eval_config(args):
    config = apply_training_recipe(Config(), args.recipe)
    if args.data_path:
        config.data_path = args.data_path
    if args.data_contract:
        config.data_contract_path = args.data_contract
    config.evaluation_split = args.split
    config.allow_legacy_checkpoint_loading = args.allow_legacy_checkpoint
    if args.seed is not None:
        config.seed = args.seed

    config.use_plotting = False
    config.use_generative_save = False
    config.pretrain_diffusion = False
    config.trainer_accelerator = args.accelerator

    config = apply_config_overrides(config, getattr(args, "config_overrides", None))
    return apply_runtime_config_overrides(config)


def build_cost_bucket_report(predictions, regression_targets):
    target_dollars = np.expm1(regression_targets)
    sorted_indices = np.argsort(target_dollars, kind="mergesort")
    bucket_names = [
        "q1_low_cost",
        "q2_mid_low_cost",
        "q3_mid_high_cost",
        "q4_high_cost",
    ]
    bucket_reports = {}
    for bucket_name, bucket_indices in zip(bucket_names, np.array_split(sorted_indices, len(bucket_names))):
        if bucket_indices.size == 0:
            continue
        bucket_metrics = score_regression_predictions(
            predictions[bucket_indices],
            regression_targets[bucket_indices],
        )
        bucket_values = target_dollars[bucket_indices]
        bucket_metrics.update(
            {
                "target_dollars_min": float(np.min(bucket_values)),
                "target_dollars_median": float(np.median(bucket_values)),
                "target_dollars_max": float(np.max(bucket_values)),
            }
        )
        bucket_reports[bucket_name] = bucket_metrics
    return bucket_reports


def save_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.accelerator)
    config = build_eval_config(args)
    pl.seed_everything(config.seed, workers=True)

    train_dataset, _, eval_dataset, _, config, dataset = prepare_data(
        config,
        requested_eval_split=args.split,
    )
    train_probe_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_eval_fn,
        shuffle=False,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_eval_fn,
        shuffle=False,
    )

    model = load_claims_model_checkpoint(
        HierarchicalClaimsModel,
        args.checkpoint,
        config=config,
        map_location=device,
        allow_legacy=config.allow_legacy_checkpoint_loading,
    )

    train_embeddings, _, train_targets, _ = collect_patient_representations(
        model,
        train_probe_dataloader,
        device=device,
        max_samples=None,
        representation_source=args.representation_source,
    )
    embeddings, _, regression_targets, _ = collect_patient_representations(
        model,
        eval_dataloader,
        device=device,
        max_samples=args.max_samples,
        representation_source=args.representation_source,
    )
    overall_metrics, predictions = compute_heldout_regression_probe_metrics(
        train_embeddings,
        train_targets,
        embeddings,
        regression_targets,
        return_predictions=True,
    )
    results = {
        **overall_metrics,
        "target_cost_bucket": build_cost_bucket_report(
            predictions,
            regression_targets,
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": str(Path(config.data_path).resolve()),
        "recipe": config.train_recipe,
        "device": str(device),
        "seed": int(config.seed),
        "representation_source": args.representation_source,
        "evaluation_split": args.split,
        "data_contract_hash": config.data_contract_hash,
        "vocab_hash": config.vocab_hash,
        "num_samples": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "selection_metric_priority": [
            "target_probe_mae_dollars",
            "target_probe_wape_percent",
            "target_probe_rmse_dollars",
            "target_cost_bucket.q4_high_cost.target_probe_rmse_dollars",
        ],
    }

    save_json(args.output_json, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
