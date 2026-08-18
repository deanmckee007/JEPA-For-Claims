import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_models.ranking import (
    PairwiseCostRankingHead,
    all_pair_indices,
    lambdarank_pair_weights,
    pairwise_logistic_loss,
    pairwise_ranknet_loss,
)
from jepa_utils.checkpointing import load_claims_model_checkpoint
from jepa_utils.config import (
    Config,
    apply_config_overrides,
    apply_runtime_config_overrides,
    apply_training_recipe,
)
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import collect_patient_representations


VARIANTS = (
    "rolled_ranknet",
    "rolled_ranknet_pointwise",
    "tail_ranknet",
    "tail_ranknet_pointwise",
    "all_ranknet",
    "lambdarank",
    "lambdarank_pointwise",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen-representation cost-ranking heads on validation. "
            "Test reporting requires an explicit additional flag."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--recipe", default="composable_level1_sigreg")
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Stratified group size used by dense all-pairs variants.",
    )
    parser.add_argument(
        "--legacy-batch-size",
        type=int,
        default=256,
        help="Batch size for the historical one-rolled-partner baseline.",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-strata", type=int, default=10)
    parser.add_argument("--pointwise-weight", type=float, default=0.1)
    parser.add_argument(
        "--tail-pair-weight",
        type=float,
        default=4.0,
        help="Additional weight for rolled pairs containing a top-decile claim.",
    )
    parser.add_argument("--lambda-gain-scale", type=float, default=4.0)
    parser.add_argument("--calibration-folds", type=int, default=5)
    parser.add_argument("--max-eval-pairs", type=int, default=500000)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=VARIANTS,
        default=list(VARIANTS),
    )
    parser.add_argument(
        "--report-split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--allow-test-report",
        action="store_true",
        help="Required with --report-split test to prevent accidental test reuse.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        help="Config override in key=value form. Repeat to set multiple fields.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-head", required=True)
    args = parser.parse_args(argv)
    if args.report_split == "test" and not args.allow_test_report:
        parser.error("--report-split test requires --allow-test-report")
    if args.group_size < 2 or args.legacy_batch_size < 2:
        parser.error("ranking group sizes must be at least 2")
    if args.num_strata < 2:
        parser.error("--num-strata must be at least 2")
    if args.calibration_folds < 2:
        parser.error("--calibration-folds must be at least 2")
    if args.pointwise_weight < 0.0:
        parser.error("--pointwise-weight must be non-negative")
    if args.tail_pair_weight < 0.0:
        parser.error("--tail-pair-weight must be non-negative")
    return args


def resolve_device(accelerator):
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_eval_loader(subset, dataset, batch_size):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )


def empirical_percentiles(targets):
    order = np.argsort(targets, kind="mergesort")
    percentiles = np.empty(len(targets), dtype=np.float32)
    if len(targets) <= 1:
        percentiles.fill(1.0)
    else:
        percentiles[order] = np.linspace(0.0, 1.0, len(targets), dtype=np.float32)
    return percentiles


def stratified_epoch_batches(targets, batch_size, num_strata, seed, epoch):
    """Use each sample once while mixing target quantiles into every group."""
    targets = np.asarray(targets)
    num_batches = max(1, math.ceil(len(targets) / batch_size))
    sorted_indices = np.argsort(targets, kind="mergesort")
    strata = np.array_split(sorted_indices, min(num_strata, len(targets)))
    rng = np.random.default_rng(seed + epoch)
    batch_members = [[] for _ in range(num_batches)]
    for stratum in strata:
        shuffled = np.array(stratum, copy=True)
        rng.shuffle(shuffled)
        start = int(rng.integers(0, num_batches))
        for offset, sample_index in enumerate(shuffled.tolist()):
            batch_members[(start + offset) % num_batches].append(sample_index)
    for members in batch_members:
        rng.shuffle(members)
    return [np.asarray(members, dtype=np.int64) for members in batch_members if members]


def stratified_calibration_split(targets, fraction, num_strata, seed):
    """Return disjoint calibration/scoring indices balanced by target quantile."""
    sorted_indices = np.argsort(targets, kind="mergesort")
    strata = np.array_split(sorted_indices, min(num_strata, len(targets)))
    rng = np.random.default_rng(seed)
    calibration = []
    scoring = []
    for stratum in strata:
        shuffled = np.array(stratum, copy=True)
        rng.shuffle(shuffled)
        count = min(len(shuffled) - 1, max(1, int(round(len(shuffled) * fraction))))
        calibration.extend(shuffled[:count].tolist())
        scoring.extend(shuffled[count:].tolist())
    return (
        np.asarray(sorted(calibration), dtype=np.int64),
        np.asarray(sorted(scoring), dtype=np.int64),
    )


def stratified_fold_assignments(targets, num_folds, num_strata, seed):
    """Assign every item to one balanced out-of-fold calibration fold."""
    sorted_indices = np.argsort(targets, kind="mergesort")
    strata = np.array_split(sorted_indices, min(num_strata, len(targets)))
    rng = np.random.default_rng(seed)
    assignments = np.empty(len(targets), dtype=np.int64)
    for stratum in strata:
        shuffled = np.array(stratum, copy=True)
        rng.shuffle(shuffled)
        start = int(rng.integers(0, num_folds))
        for offset, sample_index in enumerate(shuffled.tolist()):
            assignments[sample_index] = (start + offset) % num_folds
    return assignments


def train_pairwise_head(
    embeddings,
    targets,
    *,
    variant,
    hidden_dim,
    epochs,
    group_size,
    legacy_batch_size,
    lr,
    num_strata,
    pointwise_weight,
    tail_pair_weight,
    lambda_gain_scale,
    seed,
    device,
):
    mean = embeddings.mean(axis=0).astype(np.float32)
    std = np.clip(embeddings.std(axis=0).astype(np.float32), 1e-6, None)
    standardized = ((embeddings - mean) / std).astype(np.float32)
    float_targets = targets.astype(np.float32)
    relevance = empirical_percentiles(float_targets)
    target_mean = float(float_targets.mean())
    target_std = float(max(float_targets.std(), 1e-6))

    pl.seed_everything(seed, workers=True)
    head = PairwiseCostRankingHead(
        input_dim=standardized.shape[1],
        hidden_dim=hidden_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    epoch_losses = []
    epoch_pair_counts = []
    dense_pairs = variant in {"all_ranknet", "lambdarank", "lambdarank_pointwise"}
    use_lambda = variant in {"lambdarank", "lambdarank_pointwise"}
    use_pointwise = variant in {
        "rolled_ranknet_pointwise",
        "tail_ranknet_pointwise",
        "lambdarank_pointwise",
    }
    use_tail_weights = variant in {"tail_ranknet", "tail_ranknet_pointwise"}
    effective_group_size = group_size if dense_pairs else legacy_batch_size

    for epoch in range(epochs):
        head.train()
        batch_losses = []
        pair_count = 0
        batches = stratified_epoch_batches(
            float_targets,
            effective_group_size,
            num_strata,
            seed,
            epoch,
        )
        for batch_indices in batches:
            batch_embeddings = torch.from_numpy(standardized[batch_indices]).to(device)
            batch_targets = torch.from_numpy(float_targets[batch_indices]).to(device)
            batch_relevance = torch.from_numpy(relevance[batch_indices]).to(device)
            scores = head(batch_embeddings)

            if dense_pairs:
                left, right = all_pair_indices(scores.size(0), device=device)
                weights = None
                if use_lambda:
                    weights = lambdarank_pair_weights(
                        scores,
                        batch_relevance,
                        left,
                        right,
                        gain_scale=lambda_gain_scale,
                    )
                ranking_loss = pairwise_logistic_loss(
                    scores,
                    batch_targets,
                    left,
                    right,
                    pair_weights=weights,
                )
                pair_count += int(left.numel())
            else:
                partner_indices = torch.roll(
                    torch.arange(scores.size(0), device=device),
                    shifts=1,
                )
                if use_tail_weights:
                    left = torch.arange(scores.size(0), device=device)
                    pair_has_top_decile = (
                        batch_relevance.ge(0.9)
                        | batch_relevance.index_select(0, partner_indices).ge(0.9)
                    )
                    pair_weights = 1.0 + (
                        pair_has_top_decile.to(scores.dtype) * tail_pair_weight
                    )
                    ranking_loss = pairwise_logistic_loss(
                        scores,
                        batch_targets,
                        left,
                        partner_indices,
                        pair_weights=pair_weights,
                    )
                else:
                    ranking_loss = pairwise_ranknet_loss(
                        scores,
                        batch_targets,
                        partner_indices,
                    )
                pair_count += int(scores.size(0))

            loss = ranking_loss
            if use_pointwise:
                normalized_targets = (batch_targets - target_mean) / target_std
                loss = loss + pointwise_weight * F.smooth_l1_loss(
                    scores,
                    normalized_targets,
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(batch_losses)))
        epoch_pair_counts.append(pair_count)
    return {
        "head": head,
        "input_mean": mean,
        "input_std": std,
        "target_mean": target_mean,
        "target_std": target_std,
        "epoch_losses": epoch_losses,
        "epoch_pair_counts": epoch_pair_counts,
    }


def score_head(head, embeddings, mean, std, device, batch_size=1024):
    standardized = ((embeddings - mean) / std).astype(np.float32)
    chunks = []
    head.eval()
    with torch.no_grad():
        for start in range(0, standardized.shape[0], batch_size):
            batch = torch.from_numpy(standardized[start : start + batch_size]).to(device)
            chunks.append(head(batch).cpu().numpy())
    return np.concatenate(chunks)


def pairwise_accuracy(scores, targets, max_pairs, seed):
    n = len(scores)
    total_possible = n * (n - 1) // 2
    if total_possible <= max_pairs:
        left, right = np.triu_indices(n, k=1)
    else:
        rng = np.random.default_rng(seed)
        left = rng.integers(0, n, size=max_pairs)
        right = rng.integers(0, n - 1, size=max_pairs)
        right = right + (right >= left)
    target_difference = targets[left] - targets[right]
    valid = target_difference != 0
    score_difference = scores[left] - scores[right]
    correct = np.sign(score_difference[valid]) == np.sign(target_difference[valid])
    return float(np.mean(correct)), int(valid.sum()), int(total_possible)


def top_decile_recall(scores, targets):
    count = max(1, int(np.ceil(0.1 * len(scores))))
    predicted_top = set(np.argpartition(scores, -count)[-count:].tolist())
    actual_top = set(np.argpartition(targets, -count)[-count:].tolist())
    return float(len(predicted_top & actual_top) / count)


def dollar_metrics(predictions, targets):
    error = predictions - targets
    target_volume = np.abs(targets).sum()
    return {
        "mae_dollars": float(np.abs(error).mean()),
        "rmse_dollars": float(np.sqrt(np.square(error).mean())),
        "wape_percent": float(100.0 * np.abs(error).sum() / target_volume),
    }


def evaluate_variant(
    scores,
    targets,
    calibration_scores,
    calibration_targets,
    args,
    *,
    out_of_fold_calibration=False,
):
    accuracy, evaluated_pairs, total_pairs = pairwise_accuracy(
        scores,
        targets,
        args.max_eval_pairs,
        args.seed,
    )
    spearman = float(spearmanr(scores, targets).statistic)
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    if out_of_fold_calibration:
        fold_assignments = stratified_fold_assignments(
            targets,
            args.calibration_folds,
            args.num_strata,
            args.seed,
        )
        calibrated_dollars = np.empty(len(targets), dtype=np.float64)
        for fold in range(args.calibration_folds):
            heldout = fold_assignments == fold
            fit = ~heldout
            fold_calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
            fold_calibrator.fit(scores[fit], np.expm1(targets[fit]))
            calibrated_dollars[heldout] = fold_calibrator.predict(scores[heldout])
        # Persist a full-validation calibrator for later sealed-test use, while
        # reporting only leakage-free out-of-fold predictions here.
        calibrator.fit(scores, np.expm1(targets))
    else:
        calibrator.fit(calibration_scores, np.expm1(calibration_targets))
        calibrated_dollars = calibrator.predict(scores)
    return {
        "pairwise_accuracy": accuracy,
        "pairwise_pairs_evaluated": evaluated_pairs,
        "pairwise_pairs_available": total_pairs,
        "spearman_correlation": spearman,
        "top_decile_recall": top_decile_recall(scores, targets),
        "monotonic_dollar_calibration": dollar_metrics(
            calibrated_dollars,
            np.expm1(targets),
        ),
    }, calibrator


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

    train_subset, _, val_subset, _, config, dataset = prepare_data(
        config,
        requested_eval_split="val",
    )
    extraction_batch_size = max(args.legacy_batch_size, args.group_size)
    train_loader = make_eval_loader(train_subset, dataset, extraction_batch_size)
    val_loader = make_eval_loader(val_subset, dataset, extraction_batch_size)

    model = load_claims_model_checkpoint(
        HierarchicalClaimsModel,
        args.checkpoint,
        config=config,
        map_location=device,
    )
    train_embeddings, _, train_targets, _ = collect_patient_representations(
        model, train_loader, device=device, max_samples=None
    )
    val_embeddings, _, val_targets, _ = collect_patient_representations(
        model, val_loader, device=device, max_samples=None
    )

    if args.report_split == "val":
        calibration_embeddings = val_embeddings
        calibration_targets = val_targets
        scoring_embeddings = val_embeddings
        scoring_targets = val_targets
        protocol = {
            "ranking_head_fit_split": "train",
            "calibration_fit_split": (
                f"stratified_{args.calibration_folds}_fold_out_of_fold_val"
            ),
            "reported_evaluation_split": "full_frozen_val",
            "test_accessed": False,
        }
    else:
        test_indices = [
            index for index, split in enumerate(dataset.split_labels) if split == "test"
        ]
        test_subset = Subset(dataset, test_indices)
        test_loader = make_eval_loader(test_subset, dataset, extraction_batch_size)
        scoring_embeddings, _, scoring_targets, _ = collect_patient_representations(
            model, test_loader, device=device, max_samples=None
        )
        calibration_embeddings = val_embeddings
        calibration_targets = val_targets
        protocol = {
            "ranking_head_fit_split": "train",
            "calibration_fit_split": "val",
            "reported_evaluation_split": "test",
            "test_accessed": True,
            "test_access_explicitly_authorized": True,
        }

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    variant_reports = {}
    head_payloads = {}
    for variant in args.variants:
        trained = train_pairwise_head(
            train_embeddings,
            train_targets,
            variant=variant,
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            group_size=args.group_size,
            legacy_batch_size=args.legacy_batch_size,
            lr=args.lr,
            num_strata=args.num_strata,
            pointwise_weight=args.pointwise_weight,
            tail_pair_weight=args.tail_pair_weight,
            lambda_gain_scale=args.lambda_gain_scale,
            seed=args.seed,
            device=device,
        )
        calibration_scores = score_head(
            trained["head"],
            calibration_embeddings,
            trained["input_mean"],
            trained["input_std"],
            device,
        )
        scoring_scores = score_head(
            trained["head"],
            scoring_embeddings,
            trained["input_mean"],
            trained["input_std"],
            device,
        )
        metrics, calibrator = evaluate_variant(
            scoring_scores,
            scoring_targets,
            calibration_scores,
            calibration_targets,
            args,
            out_of_fold_calibration=args.report_split == "val",
        )
        metrics.update(
            {
                "training_loss_first": trained["epoch_losses"][0],
                "training_loss_last": trained["epoch_losses"][-1],
                "training_pairs_per_epoch_first": trained["epoch_pair_counts"][0],
                "training_pairs_total": int(sum(trained["epoch_pair_counts"])),
            }
        )
        variant_reports[variant] = metrics
        head_payloads[variant] = {
            "state_dict": trained["head"].cpu().state_dict(),
            "input_mean": trained["input_mean"],
            "input_std": trained["input_std"],
            "target_mean": trained["target_mean"],
            "target_std": trained["target_std"],
            "isotonic_x_thresholds": calibrator.X_thresholds_,
            "isotonic_y_thresholds": calibrator.y_thresholds_,
        }

    report = {
        "protocol": protocol,
        "num_train": int(len(train_targets)),
        "num_calibration": int(len(calibration_targets)),
        "num_reported": int(len(scoring_targets)),
        "variants": variant_reports,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_contract_hash": config.data_contract_hash,
        "vocab_hash": config.vocab_hash,
        "claim_inclusion_policy": config.claim_inclusion_policy,
        "seed": args.seed,
        "hyperparameters": {
            "epochs": args.epochs,
            "group_size": args.group_size,
            "legacy_batch_size": args.legacy_batch_size,
            "num_strata": args.num_strata,
            "pointwise_weight": args.pointwise_weight,
            "tail_pair_weight": args.tail_pair_weight,
            "lambda_gain_scale": args.lambda_gain_scale,
            "calibration_folds": args.calibration_folds,
        },
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_head = Path(args.output_head)
    output_head.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "variants": head_payloads,
            "config": vars(args),
            "protocol": protocol,
        },
        output_head,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
