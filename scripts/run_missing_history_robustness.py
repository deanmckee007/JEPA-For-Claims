"""Audit frozen tail representations under deterministic missing claim history."""

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
from jepa_utils.representation_eval import select_representation_tensor
from scripts.run_cost_attribution_ladder import parse_checkpoint_spec
from scripts.run_frozen_generation_probe import resolve_device
from scripts.run_high_cost_tail_probe import (
    TAIL_METRICS,
    balanced_sample_weights,
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
        "--drop-ratios", nargs="+", type=float,
        default=[0.0, 0.1, 0.3, 0.5, 0.7],
    )
    parser.add_argument("--boost-iterations", type=int, default=100)
    args = parser.parse_args(argv)
    if any(not 0.0 <= ratio < 1.0 for ratio in args.drop_ratios):
        parser.error("drop ratios must be in [0, 1)")
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


def drop_context_claims(cpt, icd, ttnc, *, drop_ratio, future_claim_k, generator):
    """Drop context claims jointly across modalities, retaining its latest claim.

    The held-out future suffix is never touched. This mirrors the model's
    per-patient context/future split and avoids turning a robustness audit into
    a different prediction target.
    """
    if drop_ratio == 0.0:
        return cpt, icd, ttnc, 1.0
    cpt_out = cpt.clone()
    icd_out = icd.clone()
    ttnc_out = ttnc.clone()
    valid = ttnc.ne(0)
    retained = 0
    available = 0
    future_slots = int(future_claim_k) + 1
    random_values = torch.rand(valid.shape, generator=generator)
    for row in range(valid.size(0)):
        valid_indices = torch.nonzero(valid[row], as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            continue
        available_future = min(future_slots, max(int(valid_indices.numel()) - 1, 1))
        context_indices = valid_indices[: valid_indices.numel() - available_future]
        if context_indices.numel() == 0:
            continue
        keep = random_values[row, context_indices] >= drop_ratio
        keep[-1] = True
        drop_indices = context_indices[~keep]
        cpt_out[row, drop_indices] = 0
        icd_out[row, drop_indices] = 0
        ttnc_out[row, drop_indices] = 0
        retained += int(keep.sum())
        available += int(keep.numel())
    retention = float(retained / available) if available else 1.0
    return cpt_out, icd_out, ttnc_out, retention


def collect_perturbed_features(
    model, loader, *, drop_ratio, future_claim_k, perturbation_seed, device
):
    model = model.to(device)
    model.eval()
    generator = torch.Generator().manual_seed(int(perturbation_seed))
    feature_chunks = []
    target_chunks = []
    retained_claims = 0.0
    batches = 0
    with torch.no_grad():
        for cpt, icd, ttnc, target in loader:
            cpt, icd, ttnc, retention = drop_context_claims(
                cpt, icd, ttnc,
                drop_ratio=drop_ratio,
                future_claim_k=future_claim_k,
                generator=generator,
            )
            outputs = model(
                cpt_tensor=cpt.to(device),
                icd_tensor=icd.to(device),
                ttnc_tensor=ttnc.to(device),
                target=target.to(device),
                teacher_forcing=True,
                generation=False,
            )
            feature_chunks.append(
                select_representation_tensor(
                    outputs, "patient_representation_pre_sae"
                ).detach().cpu().numpy()
            )
            target_chunks.append(target.numpy())
            retained_claims += retention
            batches += 1
    return (
        np.concatenate(feature_chunks).astype(np.float32),
        np.concatenate(target_chunks).astype(np.float32),
        retained_claims / max(batches, 1),
    )


def fit_boosted(train_x, labels, *, seed, iterations):
    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=iterations,
        max_leaf_nodes=15, min_samples_leaf=5, l2_regularization=2.0,
        early_stopping=False, random_state=seed,
    )
    model.fit(train_x, labels, sample_weight=balanced_sample_weights(labels))
    return model


def fit_logistic(train_x, labels, *, seed):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000,
            solver="liblinear", random_state=seed,
        ),
    )
    model.fit(train_x, labels)
    return model


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["condition"], run["drop_ratio"])].append(run)
    output = {}
    for (condition, ratio), rows in sorted(grouped.items()):
        metrics = {}
        for metric in TAIL_METRICS:
            values = np.asarray([row["metrics"][metric] for row in rows])
            metrics[metric] = {
                "mean": float(values.mean()),
                "sample_std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            }
        output.setdefault(condition, {})[str(ratio)] = {
            "num_runs": len(rows),
            "context_retention_mean": float(
                np.mean([row["context_retention"] for row in rows])
            ),
            "metrics": metrics,
        }
    for condition, ratios in output.items():
        baseline = ratios[str(0.0)]["metrics"]
        for row in ratios.values():
            row["delta_from_no_drop"] = {
                metric: row["metrics"][metric]["mean"] - baseline[metric]["mean"]
                for metric in TAIL_METRICS
            }
    return output


def render_markdown(aggregate, ratios):
    lines = [
        "# Missing-History Robustness", "",
        "Each cell is mean precision / PR-AUC across encoder seeds. The tail head is fit once on complete training histories.",
        "",
        "| Condition | " + " | ".join(f"drop {100*r:g}%" for r in ratios) + " |",
        "|---|" + "---:|" * len(ratios),
    ]
    for condition, rows in aggregate.items():
        cells = []
        for ratio in ratios:
            metrics = rows[str(ratio)]["metrics"]
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
    labels = None
    threshold = None
    runs = []
    ratios = sorted(set(args.drop_ratios))
    for checkpoint in args.checkpoints:
        group = checkpoint["group"]
        seed = checkpoint["seed"]
        config = configure(checkpoint["path"], args)
        pl.seed_everything(seed, workers=True)
        model = load_claims_model_checkpoint(
            HierarchicalClaimsModel, checkpoint["path"], config=config,
            map_location=device,
        )
        train_x, train_targets, _ = collect_perturbed_features(
            model, train_loader, drop_ratio=0.0,
            future_claim_k=getattr(config, "future_claim_k", 0),
            perturbation_seed=seed, device=device,
        )
        if labels is None:
            labels = top_fraction_labels(train_targets, args.tail_fraction)
            threshold = train_tail_threshold(train_targets, args.tail_fraction)
        boosted = fit_boosted(
            train_x, labels, seed=seed, iterations=args.boost_iterations
        )
        logistic = fit_logistic(train_x, labels, seed=seed)
        # Fit once: changing scores below reflects representation robustness,
        # not adaptation of the downstream head to missingness.
        for ratio in ratios:
            val_x, val_targets, retention = collect_perturbed_features(
                model, val_loader, drop_ratio=ratio,
                future_claim_k=getattr(config, "future_claim_k", 0),
                perturbation_seed=10_000 + seed + int(round(ratio * 1_000)),
                device=device,
            )
            boosted_scores = boosted.predict_proba(val_x)[:, 1]
            logistic_scores = logistic.predict_proba(val_x)[:, 1]
            for learner, scores in (
                ("boosted", boosted_scores), ("logistic", logistic_scores)
            ):
                runs.append({
                    "condition": f"{group}_{learner}",
                    "seed": int(seed),
                    "drop_ratio": float(ratio),
                    "context_retention": float(retention),
                    "metrics": score_tail_ranking(
                        scores, val_targets, fraction=args.tail_fraction,
                        train_threshold=threshold,
                    ),
                })
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "tail_fraction": args.tail_fraction,
            "drop_ratios": ratios,
            "latest_context_claim_always_retained": True,
            "future_claims_untouched": True,
            "tail_head_fit_on_complete_training_history": True,
            "evaluation_weights": "online",
            "data_contract_hash": first_config.data_contract_hash,
            "vocab_hash": first_config.vocab_hash,
        },
        "aggregate": aggregate,
        "runs": runs,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = render_markdown(aggregate, ratios)
    (output_dir / "summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
