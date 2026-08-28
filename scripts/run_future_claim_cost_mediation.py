"""Test whether predicted future claims mediate frozen-embedding cost value."""

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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint, read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import score_regression_predictions
from scripts.run_cost_supervision_ablation import stratified_fraction_positions
from scripts.run_frozen_generation_probe import (
    collect_frozen_dataset,
    resolve_device,
    token_ids_to_multi_hot,
    write_json,
)
from scripts.run_generation_aux_cost_probe import (
    SharedCostGenerationHead,
    auxiliary_loss,
    initialize_generation_priors,
)


class CostProbe(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, linear=False):
        super().__init__()
        if linear:
            self.network = nn.Linear(input_dim, 1)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-epochs", type=int, default=20)
    parser.add_argument("--probe-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--cost-label-fractions", nargs="+", type=float,
        default=[0.01, 0.05, 0.1, 0.25],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--decoder-seed", type=int, default=42)
    parser.add_argument(
        "--claim-mediator-dims",
        nargs="*",
        type=int,
        default=[],
        help=(
            "Optional hidden widths for nonlinear predicted/oracle claim-set "
            "cost mediators. The existing flat linear controls are always retained."
        ),
    )
    args = parser.parse_args(argv)
    if any(not 0.0 < value <= 1.0 for value in args.cost_label_fractions):
        parser.error("--cost-label-fractions must be in (0, 1]")
    if any(value <= 0 for value in args.claim_mediator_dims):
        parser.error("--claim-mediator-dims must contain positive integers")
    args.claim_mediator_dims = sorted(set(args.claim_mediator_dims))
    return args


def fit_future_decoder(
    train_features,
    train_data,
    *,
    cpt_vocab_size,
    icd_vocab_size,
    ttnc_vocab_size,
    hidden_dim,
    batch_size,
    epochs,
    lr,
    weight_decay,
    seed,
    device,
):
    pl.seed_everything(seed, workers=True)
    feature_mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = np.maximum(
        train_features.std(axis=0, keepdims=True), 1e-6
    ).astype(np.float32)
    normalized = ((train_features - feature_mean) / feature_std).astype(np.float32)
    cpt = token_ids_to_multi_hot(train_data["cpt_ids"], cpt_vocab_size)
    icd = token_ids_to_multi_hot(train_data["icd_ids"], icd_vocab_size)
    ttnc = np.asarray(train_data["ttnc"], dtype=np.int64)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(normalized), torch.from_numpy(cpt),
            torch.from_numpy(icd), torch.from_numpy(ttnc),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 11),
    )
    decoder = SharedCostGenerationHead(
        normalized.shape[1], hidden_dim,
        cpt_vocab_size, icd_vocab_size, ttnc_vocab_size,
    ).to(device)
    initialize_generation_priors(decoder, cpt, icd, ttnc)
    optimizer = torch.optim.AdamW(
        decoder.parameters(), lr=lr, weight_decay=weight_decay
    )
    history = []
    for _ in range(epochs):
        decoder.train()
        losses = []
        for features, cpt_target, icd_target, ttnc_target in loader:
            features = features.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = decoder(features)
            loss = auxiliary_loss(
                outputs,
                cpt_target.to(device),
                icd_target.to(device),
                ttnc_target.to(device),
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    return decoder, feature_mean, feature_std, history


def decoder_features_and_metrics(
    decoder, features, mean, std, targets, device, batch_size=512
):
    normalized = ((features - mean) / std).astype(np.float32)
    hidden_chunks = []
    prediction_chunks = []
    cpt_chunks = []
    icd_chunks = []
    ttnc_chunks = []
    decoder.eval()
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start:start + batch_size]).to(device)
            hidden = decoder.shared_features(batch)
            outputs = decoder(batch)
            cpt_probability = torch.sigmoid(outputs["cpt_logits"][:, 1:])
            icd_probability = torch.sigmoid(outputs["icd_logits"][:, 1:])
            ttnc_logits = outputs["ttnc_logits"].clone()
            ttnc_logits[:, 0] = torch.finfo(ttnc_logits.dtype).min
            ttnc_probability = torch.softmax(ttnc_logits, dim=1)[:, 1:]
            hidden_chunks.append(hidden.cpu())
            prediction_chunks.append(torch.cat(
                [cpt_probability, icd_probability, ttnc_probability], dim=1
            ).cpu())
            cpt_chunks.append(cpt_probability.cpu())
            icd_chunks.append(icd_probability.cpu())
            ttnc_chunks.append(ttnc_logits.argmax(dim=1).cpu())
    cpt_target = token_ids_to_multi_hot(
        targets["cpt_ids"], decoder.cpt_head.out_features
    )[:, 1:]
    icd_target = token_ids_to_multi_hot(
        targets["icd_ids"], decoder.icd_head.out_features
    )[:, 1:]
    cpt_probability = torch.cat(cpt_chunks).numpy()
    icd_probability = torch.cat(icd_chunks).numpy()
    metrics = {
        "cpt_micro_average_precision": float(average_precision_score(
            cpt_target.ravel(), cpt_probability.ravel()
        )),
        "icd_micro_average_precision": float(average_precision_score(
            icd_target.ravel(), icd_probability.ravel()
        )),
        "ttnc_accuracy": float(np.mean(
            torch.cat(ttnc_chunks).numpy() == targets["ttnc"]
        )),
    }
    return (
        torch.cat(hidden_chunks).numpy().astype(np.float32),
        torch.cat(prediction_chunks).numpy().astype(np.float32),
        metrics,
    )


def oracle_claim_features(targets, cpt_vocab_size, icd_vocab_size, ttnc_vocab_size):
    cpt = token_ids_to_multi_hot(targets["cpt_ids"], cpt_vocab_size)[:, 1:]
    icd = token_ids_to_multi_hot(targets["icd_ids"], icd_vocab_size)[:, 1:]
    ttnc = np.eye(ttnc_vocab_size, dtype=np.float32)[targets["ttnc"]][:, 1:]
    return np.concatenate([cpt, icd, ttnc], axis=1).astype(np.float32)


def standardize_pair(train_features, val_features):
    mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((val_features - mean) / std).astype(np.float32),
    )


def fit_cost_probe(
    train_features,
    val_features,
    train_cost,
    val_cost,
    positions,
    *,
    linear,
    hidden_dim,
    batch_size,
    epochs,
    lr,
    weight_decay,
    steps_per_epoch,
    seed,
    device,
):
    pl.seed_everything(seed, workers=True)
    cost_mean = float(np.mean(train_cost[positions]))
    cost_std = float(max(np.std(train_cost[positions]), 1e-6))
    normalized_cost = ((train_cost - cost_mean) / cost_std).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_features[positions]),
            torch.from_numpy(normalized_cost[positions]),
        ),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed + 23),
    )
    probe = CostProbe(train_features.shape[1], hidden_dim=hidden_dim, linear=linear).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    iterator = iter(loader)
    history = []
    for _ in range(epochs):
        probe.train()
        losses = []
        for _ in range(steps_per_epoch):
            try:
                features, target = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                features, target = next(iterator)
            optimizer.zero_grad(set_to_none=True)
            prediction = probe(features.to(device))
            loss = F.mse_loss(prediction, target.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))
    probe.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(val_features), 1024):
            batch = torch.from_numpy(val_features[start:start + 1024]).to(device)
            predictions.append(probe(batch).cpu())
    predicted_log_cost = torch.cat(predictions).numpy() * cost_std + cost_mean
    return score_regression_predictions(predicted_log_cost, val_cost), history


def aggregate(runs):
    grouped = defaultdict(list)
    direct = {}
    for row in runs:
        grouped[(row["cost_label_fraction"], row["pathway"])].append(row)
        if row["pathway"] == "direct_embedding_mlp":
            direct[(row["cost_label_fraction"], row["seed"])] = row
    result = {}
    for (fraction, pathway), rows in grouped.items():
        maes = [row["cost"]["target_probe_mae_dollars"] for row in rows]
        rmses = [row["cost"]["target_probe_rmse_dollars"] for row in rows]
        deltas = [
            row["cost"]["target_probe_mae_dollars"]
            - direct[(fraction, row["seed"])]["cost"]["target_probe_mae_dollars"]
            for row in rows
        ]
        result[f"{fraction:g}|{pathway}"] = {
            "num_runs": len(rows),
            "mae_dollars": {
                "mean": float(np.mean(maes)),
                "sample_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
            },
            "rmse_dollars": {
                "mean": float(np.mean(rmses)),
                "sample_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
            },
            "paired_mae_delta_vs_direct_mlp": {
                "mean": float(np.mean(deltas)),
                "sample_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                "wins": int(sum(delta < 0 for delta in deltas)),
            },
        }
    return result


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.accelerator)
    config = copy.deepcopy(read_checkpoint_config(args.checkpoint))
    if config is None:
        raise ValueError("Checkpoint does not contain a saved Config object")
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.evaluation_split = "val"
    config.use_generative_save = False
    config.use_plotting = False
    config.pretrain_diffusion = False
    config.trainer_accelerator = args.accelerator
    config = apply_runtime_config_overrides(config)
    train_dataset, _, val_dataset, _, config, dataset = prepare_data(
        config, requested_eval_split="val"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )
    encoder = load_claims_model_checkpoint(
        HierarchicalClaimsModel, args.checkpoint, config=config, map_location=device
    )
    train_data = collect_frozen_dataset(
        encoder, train_loader, ["next_claim_prediction"], device
    )
    val_data = collect_frozen_dataset(
        encoder, val_loader, ["next_claim_prediction"], device
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_embedding = train_data["features"]["next_claim_prediction"]
    val_embedding = val_data["features"]["next_claim_prediction"]
    decoder, decoder_mean, decoder_std, decoder_history = fit_future_decoder(
        train_embedding,
        train_data,
        cpt_vocab_size=config.cpt_vocab_size,
        icd_vocab_size=config.icd_vocab_size,
        ttnc_vocab_size=config.ttnc_vocab_size,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        epochs=args.decoder_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.decoder_seed,
        device=device,
    )
    train_hidden, train_prediction, train_generation = decoder_features_and_metrics(
        decoder, train_embedding, decoder_mean, decoder_std, train_data, device
    )
    val_hidden, val_prediction, val_generation = decoder_features_and_metrics(
        decoder, val_embedding, decoder_mean, decoder_std, val_data, device
    )
    del decoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    train_oracle = oracle_claim_features(
        train_data, config.cpt_vocab_size, config.icd_vocab_size, config.ttnc_vocab_size
    )
    val_oracle = oracle_claim_features(
        val_data, config.cpt_vocab_size, config.icd_vocab_size, config.ttnc_vocab_size
    )
    embedding_pair = standardize_pair(train_embedding, val_embedding)
    hidden_pair = standardize_pair(train_hidden, val_hidden)
    prediction_pair = standardize_pair(train_prediction, val_prediction)
    oracle_pair = standardize_pair(train_oracle, val_oracle)
    source_pairs = {
        "direct_embedding_mlp": (*embedding_pair, False, args.hidden_dim),
        "direct_embedding_linear": (*embedding_pair, True, args.hidden_dim),
        "decoder_hidden_mlp": (*hidden_pair, False, args.hidden_dim),
        "predicted_claims_linear": (*prediction_pair, True, args.hidden_dim),
        "oracle_claims_linear": (*oracle_pair, True, args.hidden_dim),
    }
    for mediator_dim in args.claim_mediator_dims:
        source_pairs[f"direct_embedding_mlp_d{mediator_dim}"] = (
            *embedding_pair, False, mediator_dim,
        )
        source_pairs[f"predicted_claims_mlp_d{mediator_dim}"] = (
            *prediction_pair, False, mediator_dim,
        )
        source_pairs[f"oracle_claims_mlp_d{mediator_dim}"] = (
            *oracle_pair, False, mediator_dim,
        )
    del train_prediction, val_prediction, train_oracle, val_oracle, train_hidden, val_hidden

    steps_per_epoch = math.ceil(len(train_embedding) / args.batch_size)
    runs = []
    for fraction in sorted(set(args.cost_label_fractions)):
        for seed in args.seeds:
            positions = stratified_fraction_positions(train_data["cost"], fraction, seed)
            for pathway, (
                train_features, val_features, linear, probe_hidden_dim
            ) in source_pairs.items():
                print(
                    f"Training fraction={fraction:g} seed={seed} pathway={pathway} "
                    f"labels={len(positions)}"
                )
                cost, history = fit_cost_probe(
                    train_features,
                    val_features,
                    train_data["cost"],
                    val_data["cost"],
                    positions,
                    linear=linear,
                    hidden_dim=probe_hidden_dim,
                    batch_size=args.batch_size,
                    epochs=args.probe_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    steps_per_epoch=steps_per_epoch,
                    seed=seed,
                    device=device,
                )
                row = {
                    "cost_label_fraction": float(fraction),
                    "seed": int(seed),
                    "pathway": pathway,
                    "linear_cost_head": bool(linear),
                    "probe_hidden_dim": int(probe_hidden_dim),
                    "feature_dimension": int(train_features.shape[1]),
                    "cost": cost,
                    "training_first": history[0],
                    "training_last": history[-1],
                }
                runs.append(row)
                write_json(
                    output_dir / f"fraction_{fraction:g}__{pathway}__seed{seed}.json",
                    row,
                )

    summary = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_contract_hash": config.data_contract_hash,
            "vocab_hash": config.vocab_hash,
            "decoder_seed": args.decoder_seed,
            "decoder_epochs": args.decoder_epochs,
            "probe_epochs": args.probe_epochs,
            "cost_label_fractions": sorted(set(args.cost_label_fractions)),
            "generation_label_fraction": 1.0,
            "future_claim_targets_exclude_cost": True,
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "seeds": args.seeds,
            "claim_mediator_dims": args.claim_mediator_dims,
        },
        "decoder_training": {
            "first_loss": decoder_history[0],
            "last_loss": decoder_history[-1],
            "train_generation": train_generation,
            "validation_generation": val_generation,
        },
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({
        "decoder_training": summary["decoder_training"],
        "aggregate": summary["aggregate"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
