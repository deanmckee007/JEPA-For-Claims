"""Test whether frozen dense and sparse claims representations are complementary."""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

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
    aggregate_runs,
    append_run,
    fit_boosted_classifier,
    fit_embedding_logistic,
    render_markdown,
    top_fraction_labels,
    train_tail_threshold,
)


class MatchedFusionTailNetwork(nn.Module):
    """Equal-parameter concat projection or elementwise dense/sparse gate."""

    def __init__(self, raw_dim, embedding_dim, *, mode, hidden_dim=64):
        super().__init__()
        if mode not in {"concat", "gate"}:
            raise ValueError("mode must be concat or gate")
        self.mode = mode
        self.fusion = nn.Linear(2 * embedding_dim, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(raw_dim + embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def fused_embedding(self, dense, sparse):
        joined = torch.cat([dense, sparse], dim=1)
        if self.mode == "concat":
            return torch.nn.functional.gelu(self.fusion(joined)), None
        gate = torch.sigmoid(self.fusion(joined))
        return gate * dense + (1.0 - gate) * sparse, gate

    def forward(self, raw, dense, sparse):
        fused, gate = self.fused_embedding(dense, sparse)
        return self.head(torch.cat([raw, fused], dim=1)).squeeze(1), gate


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", dest="checkpoints", action="append",
        type=parse_checkpoint_spec, required=True,
        help="Checkpoint as dense_seedN=PATH or sparse_seedN=PATH.",
    )
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tail-fraction", type=float, default=0.015)
    parser.add_argument("--raw-top-columns", type=int, default=256)
    parser.add_argument("--boost-iterations", type=int, default=200)
    parser.add_argument("--neural-epochs", type=int, default=100)
    parser.add_argument("--neural-patience", type=int, default=10)
    args = parser.parse_args(argv)
    groups = {item["group"] for item in args.checkpoints}
    if groups != {"dense", "sparse"}:
        parser.error("checkpoints must contain dense and sparse groups")
    pairs = {(item["group"], item["seed"]) for item in args.checkpoints}
    seeds = {item["seed"] for item in args.checkpoints}
    if any((group, seed) not in pairs for seed in seeds for group in groups):
        parser.error("every seed must have both dense and sparse checkpoints")
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


def standardize_from_train(train, val):
    train = np.asarray(train, dtype=np.float32)
    val = np.asarray(val, dtype=np.float32)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (val - mean) / scale


def fit_neural_fusion(
    raw_train, dense_train, sparse_train, labels,
    raw_val, dense_val, sparse_val, *, mode, seed, device,
    epochs, patience,
):
    torch.manual_seed(seed)
    raw_train, raw_val = standardize_from_train(raw_train, raw_val)
    dense_train, dense_val = standardize_from_train(dense_train, dense_val)
    sparse_train, sparse_val = standardize_from_train(sparse_train, sparse_val)
    all_positions = np.arange(len(labels))
    fit_positions, stop_positions = train_test_split(
        all_positions, test_size=0.15, random_state=seed, stratify=labels
    )
    tensors = TensorDataset(
        torch.from_numpy(raw_train[fit_positions]),
        torch.from_numpy(dense_train[fit_positions]),
        torch.from_numpy(sparse_train[fit_positions]),
        torch.from_numpy(labels[fit_positions].astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(tensors, batch_size=512, shuffle=True, generator=generator)
    model = MatchedFusionTailNetwork(
        raw_train.shape[1], dense_train.shape[1], mode=mode
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    fit_labels = labels[fit_positions]
    positive_weight = float((fit_labels == 0).sum() / max((fit_labels == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    stop_tensors = tuple(
        torch.from_numpy(array[stop_positions]).to(device)
        for array in (raw_train, dense_train, sparse_train)
    )
    val_tensors = tuple(
        torch.from_numpy(array).to(device)
        for array in (raw_val, dense_val, sparse_val)
    )
    best_ap = -np.inf
    best_state = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(epochs):
        model.train()
        for raw_batch, dense_batch, sparse_batch, label_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(
                raw_batch.to(device), dense_batch.to(device), sparse_batch.to(device)
            )
            loss = criterion(logits, label_batch.to(device))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            stop_logits, _ = model(*stop_tensors)
            stop_scores = torch.sigmoid(stop_logits).cpu().numpy()
        stop_ap = average_precision_score(labels[stop_positions], stop_scores)
        if stop_ap > best_ap + 1e-6:
            best_ap = float(stop_ap)
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch + 1
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, gate = model(*val_tensors)
        scores = torch.sigmoid(logits).cpu().numpy()
    metadata = {"best_epoch": best_epoch, "inner_validation_ap": best_ap}
    if gate is not None:
        gate_array = gate.cpu().numpy()
        metadata.update({
            "gate_dense_weight_mean": float(gate_array.mean()),
            "gate_dense_weight_std": float(gate_array.std()),
        })
    return scores, metadata


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

    runs = []
    for seed in sorted({item["seed"] for item in args.checkpoints}):
        dense_train, dense_val = embeddings[("dense", seed)]
        sparse_train, sparse_val = embeddings[("sparse", seed)]
        dense_sparse_train = np.concatenate([dense_train, sparse_train], axis=1)
        dense_sparse_val = np.concatenate([dense_val, sparse_val], axis=1)

        embedding_sets = {
            "dense_embedding_logistic": (dense_train, dense_val),
            "sparse_embedding_logistic": (sparse_train, sparse_val),
            "dense_sparse_embedding_logistic": (dense_sparse_train, dense_sparse_val),
        }
        for condition, (train_x, val_x) in embedding_sets.items():
            scores = fit_embedding_logistic(train_x, train_labels, val_x, seed=seed)
            append_run(
                runs, condition, seed, scores, val_targets,
                args.tail_fraction, threshold,
            )

        boosted_sets = {
            "raw_boosted_tail": (raw_train, raw_val),
            "dense_boosted_tail": (dense_train, dense_val),
            "sparse_boosted_tail": (sparse_train, sparse_val),
            "raw_dense_boosted_tail": (
                np.concatenate([raw_train, dense_train], axis=1),
                np.concatenate([raw_val, dense_val], axis=1),
            ),
            "raw_sparse_boosted_tail": (
                np.concatenate([raw_train, sparse_train], axis=1),
                np.concatenate([raw_val, sparse_val], axis=1),
            ),
            "dense_sparse_boosted_tail": (dense_sparse_train, dense_sparse_val),
            "raw_dense_sparse_boosted_tail": (
                np.concatenate([raw_train, dense_train, sparse_train], axis=1),
                np.concatenate([raw_val, dense_val, sparse_val], axis=1),
            ),
        }
        for condition, (train_x, val_x) in boosted_sets.items():
            scores, iterations = fit_boosted_classifier(
                train_x, train_labels, val_x, seed=seed,
                iterations=args.boost_iterations,
            )
            append_run(
                runs, condition, seed, scores, val_targets,
                args.tail_fraction, threshold, iterations=iterations,
            )

        for mode in ("concat", "gate"):
            scores, metadata = fit_neural_fusion(
                raw_train, dense_train, sparse_train, train_labels,
                raw_val, dense_val, sparse_val, mode=mode, seed=seed,
                device=device, epochs=args.neural_epochs,
                patience=args.neural_patience,
            )
            append_run(
                runs, f"raw_dense_sparse_neural_{mode}", seed, scores,
                val_targets, args.tail_fraction, threshold, **metadata,
            )

    aggregate = aggregate_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "mortality_label_available": False,
            "task": "high_cost_tail_proxy_dense_sparse_fusion",
            "tail_fraction": args.tail_fraction,
            "fusion_scope": "separately_pretrained_frozen_encoders",
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
