"""Train a token-conditioned candidate decoder against copy and flat controls.

Retrieval uses frozen context embeddings and next claims from training only.
Training retrieval excludes its own row and duplicate context representations.
Validation targets are used only for final scoring and candidate recall.
"""
import argparse
import copy
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import read_checkpoint_config, load_claims_model_checkpoint
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from scripts.run_frozen_generation_probe import (
    collect_frozen_dataset, token_ids_to_multi_hot, resolve_device,
    fit_decoder, predict_decoder, persistence_prediction, score_generation, write_json,
)

SOURCE = "patient_representation_pre_sae"


def retrieve_neighbors(train_features, query_features, neighbors, *, training=False):
    if neighbors < 1 or len(train_features) < 2:
        raise ValueError("Retrieval needs positive k and at least two training rows")
    search = NearestNeighbors(metric="cosine", algorithm="brute").fit(train_features)
    count = min(len(train_features), neighbors + (32 if training else 0))
    result = []
    # Bound the temporary distance matrix, especially for leave-one-out training.
    for start in range(0, len(query_features), 256):
        distances, indices = search.kneighbors(query_features[start:start + 256], n_neighbors=count)
        for offset, (distance, index) in enumerate(zip(distances, indices)):
            if training:
                index = index[(index != start + offset) & (distance > 1e-6)]
            result.append(index[:neighbors])
    return result


def build_candidates(previous_ids, reference_targets, neighbors, max_candidates):
    if max_candidates < previous_ids.shape[1]:
        raise ValueError("Candidate budget must accommodate all previous-claim tokens")
    result = np.zeros((len(previous_ids), max_candidates), dtype=np.int64)
    for row, nearby in enumerate(neighbors):
        previous = sorted(set(int(x) for x in previous_ids[row] if x > 0))
        counts = Counter(int(x) for x in reference_targets[nearby].ravel() if x > 0)
        ordered = previous + [token for token, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])) if token not in previous]
        chosen = ordered[:max_candidates]
        result[row, :len(chosen)] = chosen
    return result


def candidate_recall(candidates, target_ids):
    covered = total = 0
    for candidate, target in zip(candidates, target_ids):
        truth = set(int(x) for x in target if x > 0)
        covered += len(truth.intersection(candidate.tolist()))
        total += len(truth)
    return float(covered / total) if total else 0.0


class CandidateDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, vocab_sizes):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(0.1))
        self.tokens = nn.ModuleList([nn.Embedding(size, hidden_dim, padding_idx=0) for size in vocab_sizes[:2]])
        self.biases = nn.ModuleList([nn.Embedding(size, 1, padding_idx=0) for size in vocab_sizes[:2]])
        self.ttnc_head = nn.Linear(hidden_dim, vocab_sizes[2])
        self.count_head = nn.Linear(hidden_dim, 2)

    def forward(self, features, candidates, previous):
        hidden = self.trunk(features)
        logits = []
        for tokens, bias, ids, copied in zip(self.tokens, self.biases, candidates, previous):
            value = (tokens(ids) * hidden[:, None, :]).sum(-1) / hidden.size(-1) ** 0.5
            logits.append(value + bias(ids).squeeze(-1) + 4.0 * copied)
        return logits, self.ttnc_head(hidden), torch.sigmoid(self.count_head(hidden))


class RetentionAdditionDecoder(CandidateDecoder):
    """Token-conditioned retention and addition heads with a shared trunk."""
    def __init__(self, input_dim, hidden_dim, vocab_sizes):
        super().__init__(input_dim, hidden_dim, vocab_sizes)
        self.retention_tokens = nn.ModuleList([nn.Embedding(size, hidden_dim, padding_idx=0) for size in vocab_sizes[:2]])
        self.retention_biases = nn.ModuleList([nn.Embedding(size, 1, padding_idx=0) for size in vocab_sizes[:2]])

    def forward(self, features, candidates, previous):
        hidden = self.trunk(features)
        logits = []
        for add, add_bias, retain, retain_bias, ids, copied in zip(
            self.tokens, self.biases, self.retention_tokens, self.retention_biases, candidates, previous):
            add_value = (add(ids) * hidden[:, None]).sum(-1) / hidden.size(-1) ** 0.5 + add_bias(ids).squeeze(-1)
            retain_value = (retain(ids) * hidden[:, None]).sum(-1) / hidden.size(-1) ** 0.5 + retain_bias(ids).squeeze(-1) + 4.0
            logits.append(torch.where(copied.bool(), retain_value, add_value))
        return logits, self.ttnc_head(hidden), torch.sigmoid(self.count_head(hidden))


def split_membership_loss(logits, targets, ids, copied):
    """Equal-weight retention/addition losses; negatives are retrieved candidates."""
    losses = []
    for group in (copied.bool(), ~copied.bool()):
        eligible = ids.ne(0) & group
        positive = eligible * targets
        negative = eligible * (1 - targets)
        losses.append((F.softplus(-logits) * positive).sum() / positive.sum().clamp_min(1)
            + (F.softplus(logits) * negative).sum() / negative.sum().clamp_min(1))
    return 0.5 * sum(losses)


def fit_candidate(train, val, train_candidates, val_candidates, config, args, device):
    pl.seed_everything(args.seed, workers=True)
    features = train["features"][SOURCE]
    mean = features.mean(0)
    std = np.maximum(features.std(0), 1e-6)
    sizes = [config.cpt_vocab_size, config.icd_vocab_size, config.ttnc_vocab_size]
    maxima = np.asarray([config.max_cpt_tokens, config.max_icd_tokens])
    labels, copied, counts = [], [], []
    for key, size, candidates in zip(["cpt", "icd"], sizes, train_candidates):
        target = token_ids_to_multi_hot(train[f"{key}_ids"], size)
        previous = token_ids_to_multi_hot(train[f"previous_{key}_ids"], size)
        labels.append(np.take_along_axis(target, candidates, axis=1))
        copied.append(np.take_along_axis(previous, candidates, axis=1))
        counts.append(target.sum(1))
    dataset = TensorDataset(*[torch.from_numpy(np.asarray(x)) for x in [
        ((features - mean) / std).astype(np.float32), *train_candidates,
        *labels, *copied, train["ttnc"].astype(np.int64),
        (np.stack(counts, 1) / maxima).astype(np.float32),
        train["previous_ttnc"].astype(np.int64),
    ]])
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed))
    separate = getattr(args, "separate_retention", False)
    decoder_class = RetentionAdditionDecoder if separate else CandidateDecoder
    model = decoder_class(features.shape[1], args.hidden_dim, sizes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for _ in range(args.epochs):
        model.train()
        losses = []
        for batch in loader:
            x, cpt, icd, y_cpt, y_icd, p_cpt, p_icd, timing, count, previous_timing = [x.to(device) for x in batch]
            logits, time_logits, count_pred = model(x, [cpt, icd], [p_cpt, p_icd])
            time_logits = time_logits + 2.0 * F.one_hot(previous_timing, sizes[2])
            time_logits[:, 0] = -1e9
            loss = F.cross_entropy(time_logits, timing) + 0.25 * F.smooth_l1_loss(count_pred, count)
            for value, target, ids, copied in zip(logits, [y_cpt, y_icd], [cpt, icd], [p_cpt, p_icd]):
                if separate:
                    loss = loss + split_membership_loss(value, target, ids, copied)
                    continue
                valid = ids.ne(0)
                positive = valid * target
                negative = valid * (1 - target)
                loss = loss + (F.softplus(-value) * positive).sum() / positive.sum().clamp_min(1)
                loss = loss + (F.softplus(value) * negative).sum() / negative.sum().clamp_min(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    if getattr(args, "candidate_artifact_path", None):
        path = Path(args.candidate_artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "feature_mean": mean,
            "feature_std": std, "input_dim": features.shape[1],
            "hidden_dim": args.hidden_dim, "vocab_sizes": sizes,
            "separate_retention": separate}, path)
    return predict_candidate(model, val, val_candidates, mean, std, sizes,
        args.batch_size, device), history


def predict_candidate(model, val, val_candidates, mean, std, sizes, batch_size, device):
    model.eval()
    outputs = {"cpt_logits": [], "icd_logits": [], "ttnc_logits": [], "cardinality_fraction": []}
    with torch.no_grad():
        for start in range(0, len(val["ttnc"]), batch_size):
            end = start + batch_size
            candidates = [torch.from_numpy(x[start:end]).to(device) for x in val_candidates]
            previous = [token_ids_to_multi_hot(val[f"previous_{key}_ids"][start:end], size) for key, size in zip(["cpt", "icd"], sizes)]
            copied = [torch.from_numpy(np.take_along_axis(p, c.cpu().numpy(), 1)).to(device) for p, c in zip(previous, candidates)]
            x = torch.from_numpy(((val["features"][SOURCE][start:end] - mean) / std).astype(np.float32)).to(device)
            logits, timing, counts = model(x, candidates, copied)
            previous_timing = torch.from_numpy(val["previous_ttnc"][start:end].astype(np.int64)).to(device)
            timing = timing + 2.0 * F.one_hot(previous_timing, sizes[2])
            for key, size, ids, value in zip(["cpt", "icd"], sizes, candidates, logits):
                dense = torch.full((len(x), size), -30.0, device=device)
                dense.scatter_(1, ids, value)
                dense[:, 0] = -30
                outputs[f"{key}_logits"].append(dense.cpu().numpy())
            outputs["ttnc_logits"].append(timing.cpu().numpy())
            outputs["cardinality_fraction"].append(counts.cpu().numpy())
    return {key: np.concatenate(value) for key, value in outputs.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ["checkpoint", "data-path", "data-contract", "output-dir"]:
        parser.add_argument(f"--{flag}", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    args = parser.parse_args(argv)
    if min(args.neighbors, args.max_candidates, args.epochs, args.batch_size) < 1:
        parser.error("Budgets, epochs and batch size must be positive")
    device = resolve_device(args.accelerator)
    config = copy.deepcopy(read_checkpoint_config(args.checkpoint))
    config.data_path, config.data_contract_path = args.data_path, args.data_contract
    config.use_generative_save = config.use_plotting = config.pretrain_diffusion = False
    config = apply_runtime_config_overrides(config)
    train_subset, _, val_subset, _, config, dataset = prepare_data(config, requested_eval_split="val")
    model = load_claims_model_checkpoint(HierarchicalClaimsModel, args.checkpoint, config=config, map_location=device)
    train, val = [collect_frozen_dataset(model, DataLoader(subset, batch_size=args.batch_size,
        collate_fn=dataset.collate_eval_fn), [SOURCE], device, max_samples=limit)
        for subset, limit in [(train_subset, args.max_train_samples), (val_subset, args.max_val_samples)]]
    del model
    neighbors = [retrieve_neighbors(train["features"][SOURCE], data["features"][SOURCE], args.neighbors, training=is_train)
        for data, is_train in [(train, True), (val, False)]]
    candidates = [[build_candidates(data[f"previous_{key}_ids"], train[f"{key}_ids"], nearby, args.max_candidates)
        for key in ["cpt", "icd"]] for data, nearby in zip([train, val], neighbors)]
    recall = {key: candidate_recall(ids, val[f"{key}_ids"]) for key, ids in zip(["cpt", "icd"], candidates[1])}
    print(f"Validation candidate recall: {recall}", flush=True)
    candidate_prediction, history = fit_candidate(train, val, *candidates, config, args, device)
    flat, mean, std, flat_history = fit_decoder(train["features"][SOURCE], train["cpt_ids"], train["icd_ids"], train["ttnc"],
        train["previous_cpt_ids"], train["previous_icd_ids"], train["previous_ttnc"],
        cpt_vocab_size=config.cpt_vocab_size, icd_vocab_size=config.icd_vocab_size, ttnc_vocab_size=config.ttnc_vocab_size,
        max_cpt_tokens=config.max_cpt_tokens, max_icd_tokens=config.max_icd_tokens, hidden_dim=args.hidden_dim,
        epochs=args.epochs, batch_size=args.batch_size, lr=1e-3, weight_decay=1e-4, count_loss_weight=0.25,
        copy_residual=True, copy_logit_boost=4.0, copy_ttnc_logit_boost=2.0, seed=args.seed, device=device)
    flat_prediction = predict_decoder(flat, val["features"][SOURCE], mean, std, device,
        previous_cpt_ids=val["previous_cpt_ids"], previous_icd_ids=val["previous_icd_ids"],
        previous_ttnc=val["previous_ttnc"], copy_residual=True)
    sizes = [config.cpt_vocab_size, config.icd_vocab_size]
    val_targets = [token_ids_to_multi_hot(val[f"{key}_ids"], size) for key, size in zip(["cpt", "icd"], sizes)]
    support = [token_ids_to_multi_hot(train[f"{key}_ids"], size).sum(0) for key, size in zip(["cpt", "icd"], sizes)]
    predictions = {"candidate": candidate_prediction, "flat_copy_residual": flat_prediction,
        "copy_only": persistence_prediction(val, config)}
    # Isolate candidate filtering from the learned token-conditioned head.
    masked_flat = {key: value.copy() for key, value in flat_prediction.items()}
    for key, ids in zip(["cpt", "icd"], candidates[1]):
        values = masked_flat[f"{key}_logits"]
        selected = np.take_along_axis(values, ids, axis=1)
        values[:] = -30.0
        np.put_along_axis(values, ids, selected, axis=1)
        values[:, 0] = -30.0
    predictions["flat_with_candidate_filter"] = masked_flat
    metrics = {name: score_generation(prediction, *val_targets, val["ttnc"],
        train_cpt_support=support[0], train_icd_support=support[1], max_cpt_tokens=config.max_cpt_tokens,
        max_icd_tokens=config.max_icd_tokens, min_class_support=10) for name, prediction in predictions.items()}
    write_json(Path(args.output_dir) / "summary.json", {
        "protocol": {**vars(args), "test_accessed": False, "evaluation_weights": "online",
            "data_contract_hash": config.data_contract_hash, "vocab_hash": config.vocab_hash,
            "train_samples": len(train["ttnc"]), "val_samples": len(val["ttnc"]),
            "retrieval_reference": "training next claims only; leave-one-out and duplicate-context exclusion for training",
            "recent_history": "last observed claim", "matched": "same embeddings, data, seed, epochs, hidden width; parameter counts differ"},
        "candidate_recall": recall, "metrics": metrics,
        "loss_history": {"candidate": history, "flat": flat_history}})
    print(metrics, flush=True)


if __name__ == "__main__":
    main()
