"""Fit frozen attentive top-tail probes over per-claim sequence states."""

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
from jepa_utils.representation_eval import collect_patient_representations
from scripts.run_cost_attribution_ladder import parse_checkpoint_spec
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


class FrozenAttentiveTailProbe(nn.Module):
    def __init__(self, embedding_dim, *, raw_dim=0, num_heads=4):
        super().__init__()
        compatible = [
            value for value in range(min(num_heads, embedding_dim), 0, -1)
            if embedding_dim % value == 0
        ]
        if not compatible:
            raise ValueError("No attention-head count divides embedding_dim")
        self.num_heads = compatible[0]
        self.query = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        nn.init.normal_(self.query, std=0.02)
        self.attention = nn.MultiheadAttention(
            embedding_dim, self.num_heads, batch_first=True
        )
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, 2 * embedding_dim),
            nn.GELU(),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim + raw_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def attentive_state(self, sequence_states, valid_token_mask):
        valid_token_mask = valid_token_mask.bool()
        safe_mask = valid_token_mask.clone()
        empty = ~safe_mask.any(dim=1)
        if empty.any():
            safe_mask[empty, 0] = True
            sequence_states = sequence_states.clone()
            sequence_states[empty, 0] = 0
        query = self.query.expand(sequence_states.size(0), -1, -1)
        attended, _ = self.attention(
            query, sequence_states, sequence_states,
            key_padding_mask=~safe_mask, need_weights=False,
        )
        state = self.attention_norm(query + attended)
        state = self.output_norm(state + self.feed_forward(state))
        return state[:, 0]

    def forward(self, sequence_states, valid_token_mask, raw_features):
        state = self.attentive_state(sequence_states, valid_token_mask)
        if raw_features.shape[1]:
            state = torch.cat([state, raw_features], dim=1)
        return self.classifier(state).squeeze(1)


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
    parser.add_argument("--probe-batch-size", type=int, default=256)
    parser.add_argument("--tail-fraction", type=float, default=0.015)
    parser.add_argument("--raw-top-columns", type=int, default=256)
    parser.add_argument("--boost-iterations", type=int, default=200)
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--probe-patience", type=int, default=8)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--probe-seeds", type=int, nargs="+", default=[42, 43, 44])
    return parser.parse_args(argv)


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


def standardize_raw(train, val):
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return (train - mean) / scale, (val - mean) / scale


def predict_in_batches(model, states, masks, raw, *, device, batch_size):
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            stop = start + batch_size
            logits = model(
                states[start:stop].to(device),
                masks[start:stop].to(device),
                raw[start:stop].to(device),
            )
            predictions.append(torch.sigmoid(logits).cpu())
    return torch.cat(predictions).numpy()


def fit_attentive_tail_probe(
    train_states, train_masks, train_labels, eval_states, eval_masks,
    *, raw_train=None, raw_eval=None, device, seed, epochs, patience,
    batch_size, learning_rate, num_heads,
):
    train_states = torch.as_tensor(train_states, dtype=torch.float32)
    train_masks = torch.as_tensor(train_masks, dtype=torch.bool)
    eval_states = torch.as_tensor(eval_states, dtype=torch.float32)
    eval_masks = torch.as_tensor(eval_masks, dtype=torch.bool)
    if raw_train is None:
        raw_train = np.empty((len(train_states), 0), dtype=np.float32)
        raw_eval = np.empty((len(eval_states), 0), dtype=np.float32)
    else:
        raw_train, raw_eval = standardize_raw(raw_train, raw_eval)
    raw_train = torch.as_tensor(raw_train, dtype=torch.float32)
    raw_eval = torch.as_tensor(raw_eval, dtype=torch.float32)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    fit_positions, stop_positions = train_test_split(
        np.arange(len(train_labels)), test_size=0.15,
        random_state=seed, stratify=train_labels,
    )
    fit_dataset = TensorDataset(
        train_states[fit_positions], train_masks[fit_positions],
        raw_train[fit_positions],
        torch.from_numpy(train_labels[fit_positions].astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        fit_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    torch.manual_seed(seed)
    model = FrozenAttentiveTailProbe(
        train_states.shape[2], raw_dim=raw_train.shape[1], num_heads=num_heads
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    fit_labels = train_labels[fit_positions]
    positive_weight = float((fit_labels == 0).sum() / max((fit_labels == 1).sum(), 1))
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(positive_weight, device=device)
    )
    stop_states = train_states[stop_positions]
    stop_masks = train_masks[stop_positions]
    stop_raw = raw_train[stop_positions]
    best_ap = -np.inf
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(epochs):
        model.train()
        for states_batch, masks_batch, raw_batch, label_batch in loader:
            logits = model(
                states_batch.to(device), masks_batch.to(device), raw_batch.to(device)
            )
            loss = criterion(logits, label_batch.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        stop_scores = predict_in_batches(
            model, stop_states, stop_masks, stop_raw,
            device=device, batch_size=batch_size,
        )
        score = average_precision_score(train_labels[stop_positions], stop_scores)
        if score > best_ap + 1e-6:
            best_ap = float(score)
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    eval_scores = predict_in_batches(
        model, eval_states, eval_masks, raw_eval,
        device=device, batch_size=batch_size,
    )
    return eval_scores, {
        "best_epoch": best_epoch,
        "inner_validation_ap": best_ap,
        "num_heads": model.num_heads,
        "raw_features_in_probe": int(raw_train.shape[1]),
    }


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
    labels = top_fraction_labels(train_targets, args.tail_fraction)
    threshold = train_tail_threshold(train_targets, args.tail_fraction)

    runs = []
    for checkpoint in args.checkpoints:
        checkpoint_seed = checkpoint["seed"]
        group = checkpoint["group"]
        config = configure(checkpoint["path"], args)
        pl.seed_everything(checkpoint_seed, workers=True)
        model = load_claims_model_checkpoint(
            HierarchicalClaimsModel, checkpoint["path"], config=config,
            map_location=device,
        )
        has_polyak = bool(
            getattr(model, "use_eval_polyak_average", False)
            and int(getattr(model, "eval_polyak_updates", torch.tensor(0)).item()) > 0
        )
        weight_modes = ["polyak", "online"] if has_polyak else ["online"]
        for weight_mode in weight_modes:
            polyak_setting = model.use_eval_polyak_average
            if weight_mode == "online":
                model.use_eval_polyak_average = False
            try:
                train_embeddings, _, model_train_targets, train_metadata = (
                    collect_patient_representations(
                        model, train_loader, device=device,
                        representation_source="patient_representation_pre_sae",
                        include_sequence_states=True,
                    )
                )
                val_embeddings, _, model_val_targets, val_metadata = (
                    collect_patient_representations(
                        model, val_loader, device=device,
                        representation_source="patient_representation_pre_sae",
                        include_sequence_states=True,
                    )
                )
            finally:
                model.use_eval_polyak_average = polyak_setting
            if not np.allclose(model_train_targets, train_targets) or not np.allclose(
                model_val_targets, val_targets
            ):
                raise ValueError("Raw and representation target order does not match")

            condition_prefix = f"{group}_{weight_mode}"
            raw_pooled_train = np.concatenate([raw_train, train_embeddings], axis=1)
            raw_pooled_val = np.concatenate([raw_val, val_embeddings], axis=1)
            for seed in sorted(set(args.probe_seeds)):
                pooled_scores = fit_embedding_logistic(
                    train_embeddings, labels, val_embeddings, seed=seed
                )
                append_run(
                    runs, f"{condition_prefix}_pooled_logistic", seed,
                    pooled_scores, val_targets, args.tail_fraction, threshold,
                    checkpoint_seed=checkpoint_seed,
                )
                raw_pooled_scores, iterations = fit_boosted_classifier(
                    raw_pooled_train, labels, raw_pooled_val,
                    seed=seed, iterations=args.boost_iterations,
                )
                append_run(
                    runs, f"{condition_prefix}_raw_pooled_boosted", seed,
                    raw_pooled_scores, val_targets, args.tail_fraction, threshold,
                    iterations=iterations, checkpoint_seed=checkpoint_seed,
                )
                for use_raw in (False, True):
                    attentive_scores, metadata = fit_attentive_tail_probe(
                        train_metadata["sequence_states"],
                        train_metadata["sequence_state_masks"], labels,
                        val_metadata["sequence_states"],
                        val_metadata["sequence_state_masks"],
                        raw_train=raw_train if use_raw else None,
                        raw_eval=raw_val if use_raw else None,
                        device=device, seed=seed, epochs=args.probe_epochs,
                        patience=args.probe_patience, batch_size=args.probe_batch_size,
                        learning_rate=args.probe_lr, num_heads=args.num_heads,
                    )
                    suffix = "raw_attentive" if use_raw else "attentive"
                    append_run(
                        runs, f"{condition_prefix}_{suffix}_tail", seed,
                        attentive_scores, val_targets, args.tail_fraction,
                        threshold, checkpoint_seed=checkpoint_seed, **metadata,
                    )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = aggregate_runs(runs)
    payload = {
        "protocol": {
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "mortality_label_available": False,
            "task": "high_cost_tail_proxy_frozen_attentive_sequence_probe",
            "tail_fraction": args.tail_fraction,
            "encoder_frozen": True,
            "early_stopping_split": "training_only_stratified_15_percent",
            "probe_seeds": sorted(set(args.probe_seeds)),
            "polyak_online_comparison": True,
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
