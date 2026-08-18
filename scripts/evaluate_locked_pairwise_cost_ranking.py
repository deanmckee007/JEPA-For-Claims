import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import spearmanr
from torch.utils.data import Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_models.ranking import PairwiseCostRankingHead
from jepa_utils.checkpointing import load_claims_model_checkpoint
from jepa_utils.config import (
    Config,
    apply_config_overrides,
    apply_runtime_config_overrides,
    apply_training_recipe,
)
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import collect_patient_representations
from scripts.evaluate_pairwise_cost_ranking import (
    dollar_metrics,
    make_eval_loader,
    pairwise_accuracy,
    resolve_device,
    score_head,
    top_decile_recall,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one already-selected pairwise ranking head on the sealed "
            "test split. This command never trains or selects a ranking head."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head-artifact", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--legacy-batch-size", type=int, default=256)
    parser.add_argument("--max-eval-pairs", type=int, default=500000)
    parser.add_argument("--allow-test-report", action="store_true")
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        help="Config override in key=value form. Repeat to set multiple fields.",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    if not args.allow_test_report:
        parser.error("sealed test evaluation requires --allow-test-report")
    return args


def apply_saved_isotonic(scores, x_thresholds, y_thresholds):
    """Apply the exact validation-fitted monotonic map with clipped tails."""
    return np.interp(
        np.asarray(scores, dtype=np.float64),
        np.asarray(x_thresholds, dtype=np.float64),
        np.asarray(y_thresholds, dtype=np.float64),
    )


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.accelerator)
    pl.seed_everything(args.seed, workers=True)

    config = apply_training_recipe(Config(), args.recipe)
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.seed = args.seed
    config.evaluation_split = "val"
    config.use_plotting = False
    config.use_generative_save = False
    config.pretrain_diffusion = False
    config = apply_config_overrides(config, args.config_overrides)
    config = apply_runtime_config_overrides(config)

    _, _, _, _, config, dataset = prepare_data(
        config,
        requested_eval_split="val",
    )
    test_indices = [
        index for index, split in enumerate(dataset.split_labels) if split == "test"
    ]
    test_subset = Subset(dataset, test_indices)
    test_loader = make_eval_loader(
        test_subset,
        dataset,
        args.legacy_batch_size,
    )

    model = load_claims_model_checkpoint(
        HierarchicalClaimsModel,
        args.checkpoint,
        config=config,
        map_location=device,
    )
    test_embeddings, _, test_targets, _ = collect_patient_representations(
        model,
        test_loader,
        device=device,
        max_samples=None,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    artifact = torch.load(
        args.head_artifact,
        map_location="cpu",
        weights_only=False,
    )
    if args.variant not in artifact.get("variants", {}):
        raise ValueError(
            f"Variant {args.variant!r} is absent from the locked head artifact."
        )
    payload = artifact["variants"][args.variant]
    input_mean = np.asarray(payload["input_mean"], dtype=np.float32)
    input_std = np.asarray(payload["input_std"], dtype=np.float32)
    state_dict = payload["state_dict"]
    hidden_dim, input_dim = state_dict["network.1.weight"].shape
    if input_dim != test_embeddings.shape[1]:
        raise ValueError(
            "Locked ranking head input dimension does not match test embeddings: "
            f"{input_dim} != {test_embeddings.shape[1]}."
        )

    head = PairwiseCostRankingHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    head.load_state_dict(state_dict, strict=True)
    head.eval()
    scores = score_head(
        head,
        test_embeddings,
        input_mean,
        input_std,
        device,
    )
    calibrated_dollars = apply_saved_isotonic(
        scores,
        payload["isotonic_x_thresholds"],
        payload["isotonic_y_thresholds"],
    )
    accuracy, evaluated_pairs, available_pairs = pairwise_accuracy(
        scores,
        test_targets,
        args.max_eval_pairs,
        args.seed,
    )
    report = {
        "protocol": {
            "ranking_head_fit_split": "train",
            "ranking_variant_selected_on": "validation",
            "calibration_fit_split": "validation",
            "reported_evaluation_split": "sealed_test",
            "ranking_head_retrained": False,
            "ranking_variant_reselected": False,
            "test_access_explicitly_authorized": True,
        },
        "variant": args.variant,
        "num_test": int(len(test_targets)),
        "pairwise_accuracy": accuracy,
        "pairwise_pairs_evaluated": evaluated_pairs,
        "pairwise_pairs_available": available_pairs,
        "spearman_correlation": float(spearmanr(scores, test_targets).statistic),
        "top_decile_recall": top_decile_recall(scores, test_targets),
        "monotonic_dollar_calibration": dollar_metrics(
            calibrated_dollars,
            np.expm1(test_targets),
        ),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "head_artifact": str(Path(args.head_artifact).resolve()),
        "data_contract_hash": config.data_contract_hash,
        "vocab_hash": config.vocab_hash,
        "claim_inclusion_policy": config.claim_inclusion_policy,
        "seed": args.seed,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
