"""Audit sparse claim-state support and its downstream information content.

This evaluation never touches the frozen test split.  It compares magnitude,
binary support, and combined next-claim features on the frozen validation split,
and relates temporal support turnover to observed code-set turnover.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint
from jepa_utils.config import Config, apply_runtime_config_overrides, apply_training_recipe
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import compute_heldout_regression_probe_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-val-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def resolve_device(accelerator):
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("GPU requested but CUDA is unavailable.")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _next_slot(tensor, observed_claim_k):
    return tensor[:, observed_claim_k] if tensor.ndim == 3 else tensor


def _last_valid(values, valid):
    indices = valid.long().sum(dim=1).clamp(min=1) - 1
    # Batches are left padded, so select by reverse search instead of count.
    indices = values.size(1) - 1 - valid.flip(1).long().argmax(dim=1)
    return values[torch.arange(values.size(0), device=values.device), indices]


def _jaccard_rows(left, right):
    intersection = (left & right).sum(dim=-1).float()
    union = (left | right).sum(dim=-1).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


def _code_jaccard(cpt_a, icd_a, cpt_b, icd_b, cpt_vocab_size):
    values = []
    for row in range(cpt_a.size(0)):
        a = {int(v) for v in cpt_a[row].tolist() if v}
        a.update(cpt_vocab_size + int(v) for v in icd_a[row].tolist() if v)
        b = {int(v) for v in cpt_b[row].tolist() if v}
        b.update(cpt_vocab_size + int(v) for v in icd_b[row].tolist() if v)
        union = a | b
        values.append(len(a & b) / len(union) if union else 1.0)
    return values


def collect(model, dataloader, device, max_samples, include_temporal=False):
    predictions, targets, labels, costs = [], [], [], []
    temporal_support_jaccard, temporal_code_jaccard = [], []
    model.eval().to(device)
    with torch.no_grad():
        for cpt, icd, ttnc, cost in dataloader:
            cpt_device, icd_device, ttnc_device = cpt.to(device), icd.to(device), ttnc.to(device)
            outputs = model(
                cpt_tensor=cpt_device,
                icd_tensor=icd_device,
                ttnc_tensor=ttnc_device,
                target=cost.to(device),
                teacher_forcing=True,
                generation=False,
            )
            prediction = _next_slot(outputs["prediction_lvl2"], model.observed_claim_k)
            target = _next_slot(outputs["target_lvl2"], model.observed_claim_k)
            predictions.append(prediction.cpu())
            targets.append(target.cpu())
            labels.append(_last_valid(ttnc, ttnc.ne(0)).cpu())
            costs.append(cost.cpu())

            if include_temporal:
                claim_states = model.encode_claims(cpt_device, icd_device, ttnc_device)
                valid = ttnc_device.ne(0)
                for row in range(cpt.size(0)):
                    indices = torch.nonzero(valid[row], as_tuple=False).squeeze(1)
                    if indices.numel() < 2:
                        continue
                    left_indices, right_indices = indices[:-1], indices[1:]
                    left_support = claim_states[row, left_indices].ne(0)
                    right_support = claim_states[row, right_indices].ne(0)
                    temporal_support_jaccard.extend(
                        _jaccard_rows(left_support, right_support).cpu().tolist()
                    )
                    temporal_code_jaccard.extend(
                        _code_jaccard(
                            cpt[row, left_indices.cpu()],
                            icd[row, left_indices.cpu()],
                            cpt[row, right_indices.cpu()],
                            icd[row, right_indices.cpu()],
                            model.cpt_vocab_size,
                        )
                    )

            if sum(chunk.size(0) for chunk in predictions) >= max_samples:
                break

    def stack(chunks):
        return torch.cat(chunks, dim=0)[:max_samples].numpy()

    return {
        "prediction": stack(predictions),
        "target": stack(targets),
        "label": stack(labels),
        "cost": stack(costs),
        "temporal_support_jaccard": np.asarray(temporal_support_jaccard),
        "temporal_code_jaccard": np.asarray(temporal_code_jaccard),
    }


def support_summary(prediction, target):
    prediction_support = prediction != 0
    target_support = target != 0
    intersection = np.logical_and(prediction_support, target_support).sum(axis=1)
    union = np.logical_or(prediction_support, target_support).sum(axis=1)
    jaccard = np.divide(intersection, union, out=np.ones_like(intersection, dtype=float), where=union > 0)
    activation_frequency = target_support.mean(axis=0)
    marginal = activation_frequency / max(activation_frequency.sum(), 1e-12)
    effective_dims = float(np.exp(-(marginal * np.log(np.clip(marginal, 1e-12, None))).sum()))
    return {
        "prediction_active_fraction": float(prediction_support.mean()),
        "target_active_fraction": float(target_support.mean()),
        "prediction_dead_dimension_fraction": float((prediction_support.mean(axis=0) == 0).mean()),
        "target_dead_dimension_fraction": float((activation_frequency == 0).mean()),
        "target_effective_active_dimensions": effective_dims,
        "prediction_target_support_jaccard_mean": float(jaccard.mean()),
        "prediction_target_support_jaccard_median": float(np.median(jaccard)),
        "prediction_target_mse": float(np.mean((prediction - target) ** 2)),
    }


def probe_ttnc(train_x, train_y, val_x, val_y, seed):
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
    )
    classifier.fit(train_x, train_y)
    prediction = classifier.predict(val_x)
    return {
        "accuracy": float(accuracy_score(val_y, prediction)),
        "macro_f1": float(f1_score(val_y, prediction, average="macro", zero_division=0)),
    }


def main(argv=None):
    args = parse_args(argv)
    pl.seed_everything(args.seed, workers=True)
    device = resolve_device(args.accelerator)
    config = apply_training_recipe(Config(), args.recipe)
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.evaluation_split = "val"
    config.seed = args.seed
    config.data_split_seed = 42
    config.use_generative_save = False
    config = apply_runtime_config_overrides(config)
    train_dataset, _, val_dataset, _, config, dataset = prepare_data(
        config, requested_eval_split="val"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )
    model = load_claims_model_checkpoint(
        HierarchicalClaimsModel,
        args.checkpoint,
        config=config,
        map_location=device,
    )
    train = collect(model, train_loader, device, args.max_train_samples)
    val = collect(model, val_loader, device, args.max_val_samples, include_temporal=True)

    feature_views = {
        "magnitude": (train["prediction"], val["prediction"]),
        "support": ((train["prediction"] != 0).astype(np.float32), (val["prediction"] != 0).astype(np.float32)),
        "support_plus_magnitude": (
            np.concatenate([train["prediction"], train["prediction"] != 0], axis=1),
            np.concatenate([val["prediction"], val["prediction"] != 0], axis=1),
        ),
    }
    probes = {}
    for name, (train_x, val_x) in feature_views.items():
        probes[name] = {
            "cost": compute_heldout_regression_probe_metrics(
                train_x, train["cost"], val_x, val["cost"]
            ),
            "ttnc": probe_ttnc(train_x, train["label"], val_x, val["label"], args.seed),
        }

    correlation = spearmanr(
        val["temporal_support_jaccard"],
        val["temporal_code_jaccard"],
    )
    result = {
        "recipe": args.recipe,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "evaluation_split": "val",
        "num_train_samples": int(train["prediction"].shape[0]),
        "num_val_samples": int(val["prediction"].shape[0]),
        "support": support_summary(val["prediction"], val["target"]),
        "temporal_support_code_jaccard_spearman": float(correlation.statistic),
        "temporal_support_code_jaccard_pvalue": float(correlation.pvalue),
        "num_temporal_pairs": int(val["temporal_support_jaccard"].size),
        "probes": probes,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
