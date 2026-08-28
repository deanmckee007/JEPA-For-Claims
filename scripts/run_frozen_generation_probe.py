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
from sklearn.metrics import average_precision_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import (
    load_claims_model_checkpoint,
    read_checkpoint_config,
)
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import compute_heldout_regression_probe_metrics
from scripts.run_cost_supervision_ablation import stratified_fraction_positions


REPRESENTATION_SOURCES = (
    "next_claim_prediction",
    "patient_representation_pre_sae",
    "patient_plus_next_claim",
    "raw_history_hash",
)
CONDITIONS = ("pretrained", "shuffled", "random_encoder")


class FrozenClaimDecoder(nn.Module):
    """Small matched decoder used for every frozen-representation condition."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        cpt_vocab_size,
        icd_vocab_size,
        ttnc_vocab_size,
        dropout=0.1,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cpt_head = nn.Linear(hidden_dim, cpt_vocab_size)
        self.icd_head = nn.Linear(hidden_dim, icd_vocab_size)
        self.ttnc_head = nn.Linear(hidden_dim, ttnc_vocab_size)
        self.cardinality_head = nn.Linear(hidden_dim, 2)

    def forward(self, representations):
        hidden = self.trunk(representations)
        return {
            "cpt_logits": self.cpt_head(hidden),
            "icd_logits": self.icd_head(hidden),
            "ttnc_logits": self.ttnc_head(hidden),
            "cardinality_fraction": torch.sigmoid(self.cardinality_head(hidden)),
        }


class AddRemoveClaimDecoder(nn.Module):
    """Predict code additions/removals while preserving the observed claim."""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        cpt_vocab_size,
        icd_vocab_size,
        ttnc_vocab_size,
        dropout=0.1,
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.cpt_add_head = nn.Linear(hidden_dim, cpt_vocab_size)
        self.cpt_remove_head = nn.Linear(hidden_dim, cpt_vocab_size)
        self.icd_add_head = nn.Linear(hidden_dim, icd_vocab_size)
        self.icd_remove_head = nn.Linear(hidden_dim, icd_vocab_size)
        self.ttnc_head = nn.Linear(hidden_dim, ttnc_vocab_size)

    def forward(self, representations):
        hidden = self.trunk(representations)
        return {
            "cpt_add_logits": self.cpt_add_head(hidden),
            "cpt_remove_logits": self.cpt_remove_head(hidden),
            "icd_add_logits": self.icd_add_head(hidden),
            "icd_remove_logits": self.icd_remove_head(hidden),
            "ttnc_logits": self.ttnc_head(hidden),
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Probe whether frozen JEPA representations can decode the held-out "
            "next claim, with matched shuffled and random-encoder controls."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--decoder-batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--count-loss-weight", type=float, default=0.25)
    parser.add_argument("--event-ranking-weight", type=float, default=0.1)
    parser.add_argument("--min-class-support", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--random-encoder-seed", type=int, default=314159)
    parser.add_argument("--label-fractions", nargs="+", type=float, default=[0.1, 1.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=REPRESENTATION_SOURCES,
        default=list(REPRESENTATION_SOURCES),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--skip-random-encoder", action="store_true")
    parser.add_argument("--copy-residual", action="store_true")
    parser.add_argument("--add-remove", action="store_true")
    parser.add_argument("--copy-logit-boost", type=float, default=4.0)
    parser.add_argument("--copy-ttnc-logit-boost", type=float, default=2.0)
    parser.add_argument(
        "--baselines-only",
        action="store_true",
        help="Score empirical-prior and last-observed-claim baselines without loading a model.",
    )
    args = parser.parse_args(argv)
    if any(not 0 < value <= 1 for value in args.label_fractions):
        parser.error("label fractions must be in (0, 1]")
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.copy_residual and args.add_remove:
        parser.error("--copy-residual and --add-remove are mutually exclusive")
    return args


def resolve_device(accelerator):
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_immediate_next_claim_targets(
    cpt_tensor,
    icd_tensor,
    ttnc_tensor,
    *,
    future_claim_k=0,
):
    """Match the model's immediate-future target without re-encoding it.

    Sequences are left padded. When multi-horizon prediction is active, the
    immediate target is the first claim in the held-out future suffix.
    """
    batch_size = ttnc_tensor.size(0)
    target_cpt = torch.zeros_like(cpt_tensor[:, 0])
    target_icd = torch.zeros_like(icd_tensor[:, 0])
    target_ttnc = torch.zeros_like(ttnc_tensor[:, 0])
    valid = ttnc_tensor.ne(0)
    future_slots = int(future_claim_k) + 1
    for batch_index in range(batch_size):
        valid_indices = torch.nonzero(valid[batch_index], as_tuple=False).squeeze(1)
        if valid_indices.numel() == 0:
            continue
        available_future = min(future_slots, max(int(valid_indices.numel()) - 1, 1))
        target_index = int(valid_indices[-available_future])
        target_cpt[batch_index] = cpt_tensor[batch_index, target_index]
        target_icd[batch_index] = icd_tensor[batch_index, target_index]
        target_ttnc[batch_index] = ttnc_tensor[batch_index, target_index]
    return target_cpt, target_icd, target_ttnc


def token_ids_to_multi_hot(token_ids, vocab_size):
    token_ids = np.asarray(token_ids, dtype=np.int64)
    output = np.zeros((token_ids.shape[0], vocab_size), dtype=np.float32)
    rows = np.repeat(np.arange(token_ids.shape[0]), token_ids.shape[1])
    columns = token_ids.reshape(-1)
    valid = columns != 0
    output[rows[valid], columns[valid]] = 1.0
    output[:, 0] = 0.0
    return output


def extract_previous_and_target_claims(
    cpt_tensor,
    icd_tensor,
    ttnc_tensor,
    *,
    future_claim_k=0,
):
    target_cpt, target_icd, target_ttnc = extract_immediate_next_claim_targets(
        cpt_tensor,
        icd_tensor,
        ttnc_tensor,
        future_claim_k=future_claim_k,
    )
    previous_cpt = torch.zeros_like(target_cpt)
    previous_icd = torch.zeros_like(target_icd)
    previous_ttnc = torch.zeros_like(target_ttnc)
    eligible = torch.zeros(ttnc_tensor.size(0), dtype=torch.bool)
    future_slots = int(future_claim_k) + 1
    for batch_index in range(ttnc_tensor.size(0)):
        valid_indices = torch.nonzero(
            ttnc_tensor[batch_index].ne(0), as_tuple=False
        ).squeeze(1)
        if valid_indices.numel() < 2:
            continue
        available_future = min(future_slots, max(int(valid_indices.numel()) - 1, 1))
        target_position = int(valid_indices.numel()) - available_future
        previous_index = int(valid_indices[target_position - 1])
        previous_cpt[batch_index] = cpt_tensor[batch_index, previous_index]
        previous_icd[batch_index] = icd_tensor[batch_index, previous_index]
        previous_ttnc[batch_index] = ttnc_tensor[batch_index, previous_index]
        eligible[batch_index] = True
    return {
        "eligible": eligible,
        "previous_cpt_ids": previous_cpt,
        "previous_icd_ids": previous_icd,
        "previous_ttnc": previous_ttnc,
        "cpt_ids": target_cpt,
        "icd_ids": target_icd,
        "ttnc": target_ttnc,
    }


def collect_raw_claim_pairs(loader, future_claim_k=0, max_samples=None):
    chunks = defaultdict(list)
    collected = 0
    for cpt_tensor, icd_tensor, ttnc_tensor, cost_target in loader:
        batch = extract_previous_and_target_claims(
            cpt_tensor,
            icd_tensor,
            ttnc_tensor,
            future_claim_k=future_claim_k,
        )
        eligible = batch.pop("eligible")
        for key, value in batch.items():
            chunks[key].append(value[eligible])
        chunks["cost"].append(cost_target[eligible])
        collected += int(eligible.sum())
        if max_samples is not None and collected >= max_samples:
            break
    limit = max_samples or collected
    return {
        key: torch.cat(values, dim=0)[:limit].numpy()
        for key, values in chunks.items()
    }


def hashed_raw_history_features(
    cpt_tensor,
    icd_tensor,
    ttnc_tensor,
    *,
    future_claim_k=0,
    output_dim=128,
    recency_decay=0.8,
):
    """Fixed-width signed hash of history only; the held-out target is excluded."""
    if output_dim < 8:
        raise ValueError("output_dim must be at least 8")
    features = np.zeros((ttnc_tensor.size(0), output_dim), dtype=np.float32)
    hash_width = output_dim - 4
    future_slots = int(future_claim_k) + 1

    def add_hashed(row, token, salt, value):
        mixed = (int(token) * 2654435761 + int(salt) * 2246822519) & 0xFFFFFFFF
        index = 4 + mixed % hash_width
        sign = 1.0 if ((mixed >> 16) & 1) else -1.0
        row[index] += sign * value

    for batch_index in range(ttnc_tensor.size(0)):
        valid_indices = torch.nonzero(
            ttnc_tensor[batch_index].ne(0), as_tuple=False
        ).squeeze(1)
        if valid_indices.numel() < 2:
            continue
        available_future = min(future_slots, max(int(valid_indices.numel()) - 1, 1))
        target_position = int(valid_indices.numel()) - available_future
        history_indices = valid_indices[:target_position].tolist()
        if not history_indices:
            continue
        row = features[batch_index]
        row[0] = len(history_indices) / max(ttnc_tensor.size(1), 1)
        cpt_count = 0
        icd_count = 0
        for history_position, claim_index in enumerate(history_indices):
            age = len(history_indices) - 1 - history_position
            recency_weight = float(recency_decay**age)
            is_last = history_position == len(history_indices) - 1
            for token in cpt_tensor[batch_index, claim_index].tolist():
                if token == 0:
                    continue
                cpt_count += 1
                add_hashed(row, token, 11, recency_weight)
                if is_last:
                    add_hashed(row, token, 101, 1.0)
            for token in icd_tensor[batch_index, claim_index].tolist():
                if token == 0:
                    continue
                icd_count += 1
                add_hashed(row, token, 23, recency_weight)
                if is_last:
                    add_hashed(row, token, 103, 1.0)
            ttnc = int(ttnc_tensor[batch_index, claim_index])
            add_hashed(row, ttnc, 37, recency_weight)
            if is_last:
                add_hashed(row, ttnc, 107, 1.0)
        row[1] = cpt_count / max(len(history_indices), 1)
        row[2] = icd_count / max(len(history_indices), 1)
        row[3] = float(ttnc_tensor[batch_index, history_indices[-1]]) / 22.0
    return features


def select_source(model, outputs, source):
    if source == "next_claim_prediction":
        return model._extract_next_claim_prediction(outputs["prediction_lvl2"])
    if source == "patient_representation_pre_sae":
        return outputs.get(
            "patient_representation_pre_sae",
            outputs["patient_representation"],
        )
    if source == "patient_plus_next_claim":
        patient = outputs.get(
            "patient_representation_pre_sae",
            outputs["patient_representation"],
        )
        next_claim = model._extract_next_claim_prediction(outputs["prediction_lvl2"])
        return torch.cat([patient, next_claim], dim=-1)
    raise ValueError(f"Unsupported representation source: {source}")


def collect_frozen_dataset(model, loader, sources, device, max_samples=None):
    model = model.to(device)
    model.eval()
    feature_chunks = {source: [] for source in sources}
    cpt_chunks = []
    icd_chunks = []
    ttnc_chunks = []
    cost_chunks = []
    collected = 0

    with torch.no_grad():
        for cpt_tensor, icd_tensor, ttnc_tensor, cost_target in loader:
            cpt_device = cpt_tensor.to(device)
            icd_device = icd_tensor.to(device)
            ttnc_device = ttnc_tensor.to(device)
            cost_device = cost_target.to(device)
            outputs = model(
                cpt_tensor=cpt_device,
                icd_tensor=icd_device,
                ttnc_tensor=ttnc_device,
                target=cost_device,
                teacher_forcing=True,
                generation=False,
            )
            # A next-claim probe requires at least one observed claim and one
            # held-out target claim. The broader any-code representation cohort
            # may legally contain single-claim patients, so exclude them here.
            eligible = ttnc_tensor.ne(0).sum(dim=1) >= 2
            for source in sources:
                if source == "raw_history_hash":
                    raw_features = hashed_raw_history_features(
                        cpt_tensor,
                        icd_tensor,
                        ttnc_tensor,
                        future_claim_k=getattr(model, "future_claim_k", 0),
                    )
                    feature_chunks[source].append(
                        torch.from_numpy(raw_features)[eligible]
                    )
                else:
                    feature_chunks[source].append(
                        select_source(model, outputs, source).detach().cpu()[eligible]
                    )
            claim_pairs = extract_previous_and_target_claims(
                cpt_tensor,
                icd_tensor,
                ttnc_tensor,
                future_claim_k=getattr(model, "future_claim_k", 0),
            )
            cpt_chunks.append(claim_pairs["cpt_ids"][eligible])
            icd_chunks.append(claim_pairs["icd_ids"][eligible])
            ttnc_chunks.append(claim_pairs["ttnc"][eligible])
            chunks_to_add = {
                "previous_cpt_ids": claim_pairs["previous_cpt_ids"][eligible],
                "previous_icd_ids": claim_pairs["previous_icd_ids"][eligible],
                "previous_ttnc": claim_pairs["previous_ttnc"][eligible],
            }
            for key, value in chunks_to_add.items():
                # Keep these alongside targets without duplicating the primary
                # chunk variables used for backward-compatible return keys.
                feature_chunks.setdefault(key, []).append(value)
            cost_chunks.append(cost_target.detach().cpu()[eligible])
            collected += int(eligible.sum())
            if max_samples is not None and collected >= max_samples:
                break

    limit = max_samples or collected
    return {
        "features": {
            source: torch.cat(chunks, dim=0)[:limit].numpy().astype(np.float32)
            for source, chunks in feature_chunks.items()
            if source in sources
        },
        "cpt_ids": torch.cat(cpt_chunks, dim=0)[:limit].numpy(),
        "icd_ids": torch.cat(icd_chunks, dim=0)[:limit].numpy(),
        "ttnc": torch.cat(ttnc_chunks, dim=0)[:limit].numpy(),
        "cost": torch.cat(cost_chunks, dim=0)[:limit].numpy(),
        "previous_cpt_ids": torch.cat(feature_chunks["previous_cpt_ids"], dim=0)[:limit].numpy(),
        "previous_icd_ids": torch.cat(feature_chunks["previous_icd_ids"], dim=0)[:limit].numpy(),
        "previous_ttnc": torch.cat(feature_chunks["previous_ttnc"], dim=0)[:limit].numpy(),
    }


def balanced_multilabel_loss(logits, targets):
    positive_count = targets.sum().clamp_min(1.0)
    negative_count = (1.0 - targets).sum().clamp_min(1.0)
    positive_loss = -(targets * F.logsigmoid(logits)).sum() / positive_count
    negative_loss = -((1.0 - targets) * F.logsigmoid(-logits)).sum() / negative_count
    return 0.5 * (positive_loss + negative_loss)


def masked_balanced_multilabel_loss(logits, targets, eligible_mask):
    eligible_mask = eligible_mask.to(logits.dtype)
    positive = eligible_mask * targets
    negative = eligible_mask * (1.0 - targets)
    positive_loss = -(positive * F.logsigmoid(logits)).sum() / positive.sum().clamp_min(1.0)
    negative_loss = -(negative * F.logsigmoid(-logits)).sum() / negative.sum().clamp_min(1.0)
    return 0.5 * (positive_loss + negative_loss)


def masked_multilabel_event_loss(logits, targets, eligible_mask, ranking_weight=0.1):
    """Calibrated event BCE with a small rare-positive ranking term."""
    eligible_mask = eligible_mask.to(logits.dtype)
    elementwise = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    calibrated = (elementwise * eligible_mask).sum() / eligible_mask.sum().clamp_min(1.0)
    if ranking_weight <= 0:
        return calibrated
    ranking = masked_balanced_multilabel_loss(logits, targets, eligible_mask)
    return calibrated + ranking_weight * ranking


def _event_prior(target, eligible):
    numerator = (target * eligible).sum(axis=0)
    denominator = eligible.sum(axis=0)
    return np.clip((numerator + 0.5) / (denominator + 1.0), 1e-5, 1 - 1e-5)


def initialize_add_remove_priors(
    decoder,
    cpt_target,
    icd_target,
    previous_cpt,
    previous_icd,
    ttnc_target,
):
    cpt_add = cpt_target * (1.0 - previous_cpt)
    cpt_remove = previous_cpt * (1.0 - cpt_target)
    icd_add = icd_target * (1.0 - previous_icd)
    icd_remove = previous_icd * (1.0 - icd_target)
    rates = {
        decoder.cpt_add_head: _event_prior(cpt_add, 1.0 - previous_cpt),
        decoder.cpt_remove_head: _event_prior(cpt_remove, previous_cpt),
        decoder.icd_add_head: _event_prior(icd_add, 1.0 - previous_icd),
        decoder.icd_remove_head: _event_prior(icd_remove, previous_icd),
    }
    ttnc_counts = np.bincount(
        np.asarray(ttnc_target, dtype=np.int64),
        minlength=decoder.ttnc_head.out_features,
    ).astype(np.float32)
    ttnc_counts[0] = 0
    ttnc_probability = (ttnc_counts + 1e-3) / (
        ttnc_counts.sum() + 1e-3 * len(ttnc_counts)
    )
    with torch.no_grad():
        for head, rate in rates.items():
            head.weight.zero_()
            head.bias.copy_(torch.from_numpy(np.log(rate / (1.0 - rate))))
        decoder.ttnc_head.weight.zero_()
        decoder.ttnc_head.bias.copy_(
            torch.from_numpy(np.log(np.clip(ttnc_probability, 1e-12, 1.0)))
        )


def add_remove_to_membership_prediction(
    change_prediction,
    previous_cpt,
    previous_icd,
    previous_ttnc,
    *,
    ttnc_copy_boost,
    max_cpt_tokens,
    max_icd_tokens,
):
    cpt_probability = torch.where(
        previous_cpt.bool(),
        1.0 - torch.sigmoid(change_prediction["cpt_remove_logits"]),
        torch.sigmoid(change_prediction["cpt_add_logits"]),
    )
    icd_probability = torch.where(
        previous_icd.bool(),
        1.0 - torch.sigmoid(change_prediction["icd_remove_logits"]),
        torch.sigmoid(change_prediction["icd_add_logits"]),
    )
    cpt_probability = cpt_probability.clamp(1e-7, 1 - 1e-7)
    icd_probability = icd_probability.clamp(1e-7, 1 - 1e-7)
    ttnc_logits = change_prediction["ttnc_logits"] + ttnc_copy_boost * F.one_hot(
        previous_ttnc,
        num_classes=change_prediction["ttnc_logits"].size(1),
    ).to(change_prediction["ttnc_logits"].dtype)
    return {
        "cpt_logits": torch.logit(cpt_probability),
        "icd_logits": torch.logit(icd_probability),
        "ttnc_logits": ttnc_logits,
        "cardinality_fraction": torch.stack(
            [
                (cpt_probability[:, 1:] >= 0.5).sum(dim=1) / max(max_cpt_tokens, 1),
                (icd_probability[:, 1:] >= 0.5).sum(dim=1) / max(max_icd_tokens, 1),
            ],
            dim=1,
        ).to(cpt_probability.dtype).clamp(0.0, 1.0),
    }


def initialize_decoder_from_priors(
    decoder,
    cpt_multi_hot,
    icd_multi_hot,
    ttnc_targets,
    cardinality,
):
    """Start every condition at the same train-only empirical distribution."""
    cpt_prevalence = np.clip(cpt_multi_hot.mean(axis=0), 1e-7, 1 - 1e-7)
    icd_prevalence = np.clip(icd_multi_hot.mean(axis=0), 1e-7, 1 - 1e-7)
    ttnc_counts = np.bincount(
        np.asarray(ttnc_targets, dtype=np.int64),
        minlength=decoder.ttnc_head.out_features,
    ).astype(np.float32)
    ttnc_counts[0] = 0
    ttnc_probabilities = (ttnc_counts + 1e-3) / (
        ttnc_counts.sum() + 1e-3 * len(ttnc_counts)
    )
    count_mean = np.clip(cardinality.mean(axis=0), 1e-5, 1 - 1e-5)
    with torch.no_grad():
        for head in (decoder.cpt_head, decoder.icd_head, decoder.ttnc_head):
            head.weight.zero_()
        decoder.cardinality_head.weight.zero_()
        decoder.cpt_head.bias.copy_(
            torch.from_numpy(np.log(cpt_prevalence / (1 - cpt_prevalence)))
        )
        decoder.icd_head.bias.copy_(
            torch.from_numpy(np.log(icd_prevalence / (1 - icd_prevalence)))
        )
        decoder.ttnc_head.bias.copy_(
            torch.from_numpy(np.log(np.clip(ttnc_probabilities, 1e-12, 1.0)))
        )
        decoder.cardinality_head.bias.copy_(
            torch.from_numpy(np.log(count_mean / (1 - count_mean)))
        )


def apply_copy_bias(
    prediction,
    previous_cpt,
    previous_icd,
    previous_ttnc,
    *,
    code_boost,
    ttnc_boost,
):
    prediction = dict(prediction)
    prediction["cpt_logits"] = prediction["cpt_logits"] + code_boost * previous_cpt
    prediction["icd_logits"] = prediction["icd_logits"] + code_boost * previous_icd
    ttnc_copy = F.one_hot(
        previous_ttnc,
        num_classes=prediction["ttnc_logits"].size(1),
    ).to(prediction["ttnc_logits"].dtype)
    prediction["ttnc_logits"] = prediction["ttnc_logits"] + ttnc_boost * ttnc_copy
    return prediction


def fit_decoder(
    train_features,
    train_cpt,
    train_icd,
    train_ttnc,
    train_previous_cpt,
    train_previous_icd,
    train_previous_ttnc,
    *,
    cpt_vocab_size,
    icd_vocab_size,
    ttnc_vocab_size,
    max_cpt_tokens,
    max_icd_tokens,
    hidden_dim,
    epochs,
    batch_size,
    lr,
    weight_decay,
    count_loss_weight,
    copy_residual,
    copy_logit_boost,
    copy_ttnc_logit_boost,
    seed,
    device,
):
    pl.seed_everything(seed, workers=True)
    feature_mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = np.maximum(
        train_features.std(axis=0, keepdims=True).astype(np.float32),
        1e-6,
    )
    normalized_features = (train_features - feature_mean) / feature_std
    cpt_multi_hot = token_ids_to_multi_hot(train_cpt, cpt_vocab_size)
    icd_multi_hot = token_ids_to_multi_hot(train_icd, icd_vocab_size)
    cardinality = np.stack(
        [
            cpt_multi_hot.sum(axis=1) / max(max_cpt_tokens, 1),
            icd_multi_hot.sum(axis=1) / max(max_icd_tokens, 1),
        ],
        axis=1,
    ).astype(np.float32)
    previous_cpt_multi_hot = token_ids_to_multi_hot(
        train_previous_cpt, cpt_vocab_size
    )
    previous_icd_multi_hot = token_ids_to_multi_hot(
        train_previous_icd, icd_vocab_size
    )
    dataset = TensorDataset(
        torch.from_numpy(normalized_features),
        torch.from_numpy(cpt_multi_hot),
        torch.from_numpy(icd_multi_hot),
        torch.from_numpy(np.asarray(train_ttnc, dtype=np.int64)),
        torch.from_numpy(cardinality),
        torch.from_numpy(previous_cpt_multi_hot),
        torch.from_numpy(previous_icd_multi_hot),
        torch.from_numpy(np.asarray(train_previous_ttnc, dtype=np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    decoder = FrozenClaimDecoder(
        normalized_features.shape[1],
        hidden_dim,
        cpt_vocab_size,
        icd_vocab_size,
        ttnc_vocab_size,
    ).to(device)
    initialize_decoder_from_priors(
        decoder,
        cpt_multi_hot,
        icd_multi_hot,
        train_ttnc,
        cardinality,
    )
    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    history = []
    for _ in range(epochs):
        decoder.train()
        batch_losses = []
        for (
            features,
            cpt_target,
            icd_target,
            ttnc_target,
            count_target,
            previous_cpt,
            previous_icd,
            previous_ttnc,
        ) in loader:
            features = features.to(device)
            cpt_target = cpt_target.to(device)
            icd_target = icd_target.to(device)
            ttnc_target = ttnc_target.to(device)
            count_target = count_target.to(device)
            previous_cpt = previous_cpt.to(device)
            previous_icd = previous_icd.to(device)
            previous_ttnc = previous_ttnc.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = decoder(features)
            if copy_residual:
                prediction = apply_copy_bias(
                    prediction,
                    previous_cpt,
                    previous_icd,
                    previous_ttnc,
                    code_boost=copy_logit_boost,
                    ttnc_boost=copy_ttnc_logit_boost,
                )
            ttnc_logits = prediction["ttnc_logits"].clone()
            ttnc_logits[:, 0] = torch.finfo(ttnc_logits.dtype).min
            loss = (
                balanced_multilabel_loss(prediction["cpt_logits"][:, 1:], cpt_target[:, 1:])
                + balanced_multilabel_loss(prediction["icd_logits"][:, 1:], icd_target[:, 1:])
                + F.cross_entropy(ttnc_logits, ttnc_target)
                + count_loss_weight
                * F.smooth_l1_loss(prediction["cardinality_fraction"], count_target)
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(batch_losses)))
    return decoder, feature_mean, feature_std, history


def predict_decoder(
    decoder,
    features,
    feature_mean,
    feature_std,
    device,
    *,
    previous_cpt_ids,
    previous_icd_ids,
    previous_ttnc,
    copy_residual=False,
    copy_logit_boost=4.0,
    copy_ttnc_logit_boost=2.0,
    batch_size=1024,
):
    decoder.eval()
    normalized = (features - feature_mean) / feature_std
    collected = defaultdict(list)
    previous_cpt = token_ids_to_multi_hot(
        previous_cpt_ids, decoder.cpt_head.out_features
    )
    previous_icd = token_ids_to_multi_hot(
        previous_icd_ids, decoder.icd_head.out_features
    )
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size]).to(device)
            prediction = decoder(batch)
            if copy_residual:
                prediction = apply_copy_bias(
                    prediction,
                    torch.from_numpy(previous_cpt[start : start + batch_size]).to(device),
                    torch.from_numpy(previous_icd[start : start + batch_size]).to(device),
                    torch.from_numpy(
                        np.asarray(previous_ttnc[start : start + batch_size], dtype=np.int64)
                    ).to(device),
                    code_boost=copy_logit_boost,
                    ttnc_boost=copy_ttnc_logit_boost,
                )
            for key, value in prediction.items():
                collected[key].append(value.cpu())
    return {
        key: torch.cat(chunks, dim=0).numpy()
        for key, chunks in collected.items()
    }


def fit_add_remove_decoder(
    train_features,
    train_cpt,
    train_icd,
    train_ttnc,
    train_previous_cpt,
    train_previous_icd,
    train_previous_ttnc,
    *,
    cpt_vocab_size,
    icd_vocab_size,
    ttnc_vocab_size,
    max_cpt_tokens,
    max_icd_tokens,
    hidden_dim,
    epochs,
    batch_size,
    lr,
    weight_decay,
    event_ranking_weight,
    ttnc_copy_boost,
    seed,
    device,
):
    pl.seed_everything(seed, workers=True)
    feature_mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    feature_std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    features = ((train_features - feature_mean) / feature_std).astype(np.float32)
    cpt_target = token_ids_to_multi_hot(train_cpt, cpt_vocab_size)
    icd_target = token_ids_to_multi_hot(train_icd, icd_vocab_size)
    previous_cpt = token_ids_to_multi_hot(train_previous_cpt, cpt_vocab_size)
    previous_icd = token_ids_to_multi_hot(train_previous_icd, icd_vocab_size)
    dataset = TensorDataset(
        torch.from_numpy(features),
        torch.from_numpy(cpt_target),
        torch.from_numpy(icd_target),
        torch.from_numpy(np.asarray(train_ttnc, dtype=np.int64)),
        torch.from_numpy(previous_cpt),
        torch.from_numpy(previous_icd),
        torch.from_numpy(np.asarray(train_previous_ttnc, dtype=np.int64)),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    decoder = AddRemoveClaimDecoder(
        features.shape[1],
        hidden_dim,
        cpt_vocab_size,
        icd_vocab_size,
        ttnc_vocab_size,
    ).to(device)
    initialize_add_remove_priors(
        decoder,
        cpt_target,
        icd_target,
        previous_cpt,
        previous_icd,
        train_ttnc,
    )
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for _ in range(epochs):
        decoder.train()
        batch_losses = []
        for (
            batch_features,
            batch_cpt,
            batch_icd,
            batch_ttnc,
            batch_previous_cpt,
            batch_previous_icd,
            batch_previous_ttnc,
        ) in loader:
            batch_features = batch_features.to(device)
            batch_cpt = batch_cpt.to(device)
            batch_icd = batch_icd.to(device)
            batch_ttnc = batch_ttnc.to(device)
            batch_previous_cpt = batch_previous_cpt.to(device)
            batch_previous_icd = batch_previous_icd.to(device)
            batch_previous_ttnc = batch_previous_ttnc.to(device)
            optimizer.zero_grad(set_to_none=True)
            change = decoder(batch_features)
            cpt_add_target = batch_cpt * (1.0 - batch_previous_cpt)
            cpt_remove_target = batch_previous_cpt * (1.0 - batch_cpt)
            icd_add_target = batch_icd * (1.0 - batch_previous_icd)
            icd_remove_target = batch_previous_icd * (1.0 - batch_icd)
            ttnc_logits = change["ttnc_logits"] + ttnc_copy_boost * F.one_hot(
                batch_previous_ttnc,
                num_classes=ttnc_vocab_size,
            ).to(change["ttnc_logits"].dtype)
            ttnc_logits[:, 0] = torch.finfo(ttnc_logits.dtype).min
            loss = (
                0.5
                * (
                    masked_multilabel_event_loss(
                        change["cpt_add_logits"][:, 1:],
                        cpt_add_target[:, 1:],
                        1.0 - batch_previous_cpt[:, 1:],
                        event_ranking_weight,
                    )
                    + masked_multilabel_event_loss(
                        change["cpt_remove_logits"][:, 1:],
                        cpt_remove_target[:, 1:],
                        batch_previous_cpt[:, 1:],
                        event_ranking_weight,
                    )
                )
                + 0.5
                * (
                    masked_multilabel_event_loss(
                        change["icd_add_logits"][:, 1:],
                        icd_add_target[:, 1:],
                        1.0 - batch_previous_icd[:, 1:],
                        event_ranking_weight,
                    )
                    + masked_multilabel_event_loss(
                        change["icd_remove_logits"][:, 1:],
                        icd_remove_target[:, 1:],
                        batch_previous_icd[:, 1:],
                        event_ranking_weight,
                    )
                )
                + F.cross_entropy(ttnc_logits, batch_ttnc)
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(batch_losses)))
    return decoder, feature_mean, feature_std, history


def predict_add_remove_decoder(
    decoder,
    features,
    feature_mean,
    feature_std,
    device,
    *,
    previous_cpt_ids,
    previous_icd_ids,
    previous_ttnc,
    max_cpt_tokens,
    max_icd_tokens,
    ttnc_copy_boost,
    batch_size=1024,
):
    decoder.eval()
    features = ((features - feature_mean) / feature_std).astype(np.float32)
    previous_cpt = token_ids_to_multi_hot(
        previous_cpt_ids, decoder.cpt_add_head.out_features
    )
    previous_icd = token_ids_to_multi_hot(
        previous_icd_ids, decoder.icd_add_head.out_features
    )
    collected = defaultdict(list)
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            change = decoder(torch.from_numpy(features[start:stop]).to(device))
            prediction = add_remove_to_membership_prediction(
                change,
                torch.from_numpy(previous_cpt[start:stop]).to(device),
                torch.from_numpy(previous_icd[start:stop]).to(device),
                torch.from_numpy(
                    np.asarray(previous_ttnc[start:stop], dtype=np.int64)
                ).to(device),
                ttnc_copy_boost=ttnc_copy_boost,
                max_cpt_tokens=max_cpt_tokens,
                max_icd_tokens=max_icd_tokens,
            )
            for key, value in prediction.items():
                collected[key].append(value.cpu())
            for key, value in change.items():
                collected[f"change_{key}"].append(value.cpu())
    return {key: torch.cat(chunks).numpy() for key, chunks in collected.items()}


def topk_multihot(scores, cardinalities):
    scores = np.asarray(scores)
    cardinalities = np.asarray(cardinalities).round().astype(np.int64)
    prediction = np.zeros_like(scores, dtype=np.float32)
    prediction[:, 0] = 0.0
    for row_index, count in enumerate(cardinalities):
        count = int(np.clip(count, 0, scores.shape[1] - 1))
        if count == 0:
            continue
        indices = np.argpartition(scores[row_index, 1:], -count)[-count:] + 1
        prediction[row_index, indices] = 1.0
    return prediction


def safe_average_precision(targets, probabilities, *, average):
    try:
        return float(average_precision_score(targets, probabilities, average=average))
    except ValueError:
        return float("nan")


def masked_event_metrics(targets, probabilities, eligible):
    mask = np.asarray(eligible, dtype=bool)[:, 1:]
    targets = np.asarray(targets, dtype=np.float32)[:, 1:][mask]
    probabilities = np.asarray(probabilities, dtype=np.float32)[:, 1:][mask]
    prediction = probabilities >= 0.5
    return {
        "prevalence": float(targets.mean()) if len(targets) else float("nan"),
        "average_precision": safe_average_precision(targets, probabilities, average="micro"),
        "f1_at_0_5": float(f1_score(targets, prediction, zero_division=0)),
    }


def score_change_events(prediction, target_cpt, target_icd, previous_cpt, previous_icd):
    result = {}
    for modality, target, previous in (
        ("cpt", target_cpt, previous_cpt),
        ("icd", target_icd, previous_icd),
    ):
        add_target = target * (1.0 - previous)
        remove_target = previous * (1.0 - target)
        add_probability = torch.sigmoid(
            torch.from_numpy(prediction[f"change_{modality}_add_logits"])
        ).numpy()
        remove_probability = torch.sigmoid(
            torch.from_numpy(prediction[f"change_{modality}_remove_logits"])
        ).numpy()
        result[modality] = {
            "add": masked_event_metrics(add_target, add_probability, 1.0 - previous),
            "remove": masked_event_metrics(remove_target, remove_probability, previous),
        }
    return result


def modality_metrics(
    targets,
    probabilities,
    predicted_counts,
    train_support,
    min_class_support,
):
    targets = np.asarray(targets, dtype=np.float32)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    targets[:, 0] = 0.0
    probabilities[:, 0] = 0.0
    predicted = topk_multihot(probabilities, predicted_counts)
    tp = float((predicted * targets).sum())
    fp = float((predicted * (1.0 - targets)).sum())
    fn = float(((1.0 - predicted) * targets).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    intersection = (predicted * targets).sum(axis=1)
    union = ((predicted + targets) > 0).sum(axis=1)
    supported = (
        (np.asarray(train_support) >= min_class_support)
        & (targets.sum(axis=0) > 0)
    )
    supported[0] = False
    clipped = np.clip(probabilities[:, 1:], 1e-7, 1 - 1e-7)
    nll = -np.mean(
        targets[:, 1:] * np.log(clipped)
        + (1 - targets[:, 1:]) * np.log(1 - clipped)
    )
    supported_macro_ap = (
        safe_average_precision(targets[:, supported], probabilities[:, supported], average="macro")
        if supported.any()
        else float("nan")
    )
    return {
        "micro_average_precision": safe_average_precision(
            targets[:, 1:].reshape(-1),
            probabilities[:, 1:].reshape(-1),
            average="micro",
        ),
        "supported_macro_average_precision": supported_macro_ap,
        "supported_class_count": int(supported.sum()),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "mean_set_jaccard": float(np.mean(intersection / np.maximum(union, 1))),
        "exact_set_match": float(np.mean(np.all(predicted == targets, axis=1))),
        "cardinality_mae": float(
            np.mean(np.abs(np.asarray(predicted_counts) - targets.sum(axis=1)))
        ),
        "binary_nll": float(nll),
    }


def score_generation(
    prediction,
    val_cpt,
    val_icd,
    val_ttnc,
    *,
    train_cpt_support,
    train_icd_support,
    max_cpt_tokens,
    max_icd_tokens,
    min_class_support,
):
    cpt_probabilities = 1.0 / (1.0 + np.exp(-np.clip(prediction["cpt_logits"], -30, 30)))
    icd_probabilities = 1.0 / (1.0 + np.exp(-np.clip(prediction["icd_logits"], -30, 30)))
    ttnc_logits = np.array(prediction["ttnc_logits"], copy=True)
    ttnc_logits[:, 0] = -1e9
    ttnc_logits -= ttnc_logits.max(axis=1, keepdims=True)
    ttnc_probabilities = np.exp(ttnc_logits)
    ttnc_probabilities /= ttnc_probabilities.sum(axis=1, keepdims=True)
    predicted_counts = prediction["cardinality_fraction"] * np.asarray(
        [max_cpt_tokens, max_icd_tokens], dtype=np.float32
    )
    ttnc_prediction = ttnc_probabilities.argmax(axis=1)
    true_class_probability = ttnc_probabilities[
        np.arange(len(val_ttnc)), np.asarray(val_ttnc, dtype=np.int64)
    ]
    supported_ttnc = np.unique(val_ttnc)
    supported_ttnc = supported_ttnc[supported_ttnc != 0]
    return {
        "cpt": modality_metrics(
            val_cpt,
            cpt_probabilities,
            predicted_counts[:, 0],
            train_cpt_support,
            min_class_support,
        ),
        "icd": modality_metrics(
            val_icd,
            icd_probabilities,
            predicted_counts[:, 1],
            train_icd_support,
            min_class_support,
        ),
        "ttnc": {
            "accuracy": float(np.mean(ttnc_prediction == val_ttnc)),
            "macro_f1": float(
                f1_score(
                    val_ttnc,
                    ttnc_prediction,
                    labels=supported_ttnc,
                    average="macro",
                    zero_division=0,
                )
            ),
            "negative_log_likelihood": float(
                -np.mean(np.log(np.clip(true_class_probability, 1e-12, 1.0)))
            ),
        },
    }


def empirical_prior_prediction(train_targets, val_size, config):
    cpt = token_ids_to_multi_hot(train_targets["cpt_ids"], config.cpt_vocab_size)
    icd = token_ids_to_multi_hot(train_targets["icd_ids"], config.icd_vocab_size)
    cpt_prevalence = np.clip(cpt.mean(axis=0), 1e-7, 1 - 1e-7)
    icd_prevalence = np.clip(icd.mean(axis=0), 1e-7, 1 - 1e-7)
    ttnc_counts = np.bincount(
        train_targets["ttnc"], minlength=config.ttnc_vocab_size
    ).astype(np.float64)
    ttnc_counts[0] = 0
    ttnc_probabilities = (ttnc_counts + 1e-3) / (ttnc_counts.sum() + 1e-3 * len(ttnc_counts))
    cardinality_fraction = np.asarray(
        [
            cpt.sum(axis=1).mean() / max(config.max_cpt_tokens, 1),
            icd.sum(axis=1).mean() / max(config.max_icd_tokens, 1),
        ],
        dtype=np.float32,
    )
    return {
        "cpt_logits": np.tile(
            np.log(cpt_prevalence / (1 - cpt_prevalence)), (val_size, 1)
        ),
        "icd_logits": np.tile(
            np.log(icd_prevalence / (1 - icd_prevalence)), (val_size, 1)
        ),
        "ttnc_logits": np.tile(np.log(np.clip(ttnc_probabilities, 1e-12, 1.0)), (val_size, 1)),
        "cardinality_fraction": np.tile(cardinality_fraction, (val_size, 1)),
    }


def persistence_prediction(val_targets, config, confidence=0.99):
    cpt = token_ids_to_multi_hot(
        val_targets["previous_cpt_ids"], config.cpt_vocab_size
    )
    icd = token_ids_to_multi_hot(
        val_targets["previous_icd_ids"], config.icd_vocab_size
    )
    low = 1.0 - confidence
    cpt_probability = cpt * confidence + (1.0 - cpt) * low
    icd_probability = icd * confidence + (1.0 - icd) * low
    ttnc_probability = np.full(
        (len(val_targets["ttnc"]), config.ttnc_vocab_size),
        low / max(config.ttnc_vocab_size - 1, 1),
        dtype=np.float32,
    )
    ttnc_probability[
        np.arange(len(ttnc_probability)), val_targets["previous_ttnc"]
    ] = confidence
    cardinality_fraction = np.stack(
        [
            cpt.sum(axis=1) / max(config.max_cpt_tokens, 1),
            icd.sum(axis=1) / max(config.max_icd_tokens, 1),
        ],
        axis=1,
    )
    return {
        "cpt_logits": np.log(cpt_probability / (1.0 - cpt_probability)),
        "icd_logits": np.log(icd_probability / (1.0 - icd_probability)),
        "ttnc_logits": np.log(np.clip(ttnc_probability, 1e-12, 1.0)),
        "cardinality_fraction": cardinality_fraction,
    }


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        key = (run["source"], run["condition"], str(run["label_fraction"]))
        grouped[key].append(run)
    aggregate = {}
    metric_paths = (
        ("generation", "cpt", "micro_average_precision"),
        ("generation", "icd", "micro_average_precision"),
        ("generation", "cpt", "micro_f1"),
        ("generation", "icd", "micro_f1"),
        ("generation", "ttnc", "accuracy"),
        ("cost", "target_probe_rmse_dollars"),
        ("cost", "target_probe_mae_dollars"),
    )
    for key, rows in grouped.items():
        summary = {"num_runs": len(rows), "metrics": {}}
        for path in metric_paths:
            values = []
            for row in rows:
                value = row
                for part in path:
                    value = value.get(part, {}) if isinstance(value, dict) else {}
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    values.append(float(value))
            if values:
                summary["metrics"][".".join(path)] = {
                    "mean": float(np.mean(values)),
                    "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
        aggregate["|".join(key)] = summary
    return aggregate


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
        config,
        requested_eval_split="val",
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
    if args.baselines_only:
        train_data = collect_raw_claim_pairs(
            train_loader,
            future_claim_k=getattr(config, "future_claim_k", 0),
            max_samples=args.max_train_samples,
        )
        val_data = collect_raw_claim_pairs(
            val_loader,
            future_claim_k=getattr(config, "future_claim_k", 0),
            max_samples=args.max_val_samples,
        )
        train_cpt = token_ids_to_multi_hot(train_data["cpt_ids"], config.cpt_vocab_size)
        train_icd = token_ids_to_multi_hot(train_data["icd_ids"], config.icd_vocab_size)
        val_cpt = token_ids_to_multi_hot(val_data["cpt_ids"], config.cpt_vocab_size)
        val_icd = token_ids_to_multi_hot(val_data["icd_ids"], config.icd_vocab_size)
        score_kwargs = {
            "train_cpt_support": train_cpt.sum(axis=0),
            "train_icd_support": train_icd.sum(axis=0),
            "max_cpt_tokens": config.max_cpt_tokens,
            "max_icd_tokens": config.max_icd_tokens,
            "min_class_support": args.min_class_support,
        }
        prior_metrics = score_generation(
            empirical_prior_prediction(train_data, len(val_data["ttnc"]), config),
            val_cpt,
            val_icd,
            val_data["ttnc"],
            **score_kwargs,
        )
        persistence_metrics = score_generation(
            persistence_prediction(val_data, config),
            val_cpt,
            val_icd,
            val_data["ttnc"],
            **score_kwargs,
        )
        payload = {
            "protocol": {
                "evaluation_split": "frozen_validation",
                "test_accessed": False,
                "train_samples": int(len(train_data["ttnc"])),
                "validation_samples": int(len(val_data["ttnc"])),
                "data_contract_hash": config.data_contract_hash,
                "vocab_hash": config.vocab_hash,
            },
            "empirical_prior": prior_metrics,
            "last_observed_claim_persistence": persistence_metrics,
        }
        write_json(output_dir / "baselines.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(f"Extracting pretrained features on {device}")
    pretrained_model = load_claims_model_checkpoint(
        HierarchicalClaimsModel,
        args.checkpoint,
        config=config,
        map_location=device,
    )
    train_data = collect_frozen_dataset(
        pretrained_model,
        train_loader,
        args.sources,
        device,
        max_samples=args.max_train_samples,
    )
    val_data = collect_frozen_dataset(
        pretrained_model,
        val_loader,
        args.sources,
        device,
        max_samples=args.max_val_samples,
    )
    del pretrained_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    random_train = None
    random_val = None
    if "random_encoder" in args.conditions and not args.skip_random_encoder:
        print("Extracting matched random-encoder features")
        pl.seed_everything(args.random_encoder_seed, workers=True)
        random_model = HierarchicalClaimsModel(copy.deepcopy(config))
        random_train = collect_frozen_dataset(
            random_model,
            train_loader,
            args.sources,
            device,
            max_samples=args.max_train_samples,
        )
        random_val = collect_frozen_dataset(
            random_model,
            val_loader,
            args.sources,
            device,
            max_samples=args.max_val_samples,
        )
        del random_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    train_cpt = token_ids_to_multi_hot(train_data["cpt_ids"], config.cpt_vocab_size)
    train_icd = token_ids_to_multi_hot(train_data["icd_ids"], config.icd_vocab_size)
    val_cpt = token_ids_to_multi_hot(val_data["cpt_ids"], config.cpt_vocab_size)
    val_icd = token_ids_to_multi_hot(val_data["icd_ids"], config.icd_vocab_size)
    prior_prediction = empirical_prior_prediction(train_data, len(val_data["ttnc"]), config)
    prior_metrics = score_generation(
        prior_prediction,
        val_cpt,
        val_icd,
        val_data["ttnc"],
        train_cpt_support=train_cpt.sum(axis=0),
        train_icd_support=train_icd.sum(axis=0),
        max_cpt_tokens=config.max_cpt_tokens,
        max_icd_tokens=config.max_icd_tokens,
        min_class_support=args.min_class_support,
    )
    write_json(output_dir / "empirical_prior.json", prior_metrics)

    runs = []
    artifact_dir = output_dir / "decoders"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for source in args.sources:
        for condition in args.conditions:
            if condition == "random_encoder" and random_train is None:
                continue
            for fraction in sorted(set(args.label_fractions)):
                for seed in args.seeds:
                    positions = stratified_fraction_positions(
                        train_data["cost"], fraction, seed
                    )
                    if condition == "random_encoder":
                        train_features = random_train["features"][source]
                        val_features = random_val["features"][source]
                    else:
                        train_features = train_data["features"][source]
                        val_features = val_data["features"][source]
                    if condition == "shuffled":
                        train_rng = np.random.default_rng(seed + 1009)
                        val_rng = np.random.default_rng(seed + 2003)
                        train_features = train_features[train_rng.permutation(len(train_features))]
                        val_features = val_features[val_rng.permutation(len(val_features))]

                    print(
                        f"Training source={source} condition={condition} "
                        f"fraction={fraction:g} seed={seed} labels={len(positions)}"
                    )
                    if args.add_remove:
                        decoder, feature_mean, feature_std, history = (
                            fit_add_remove_decoder(
                                train_features[positions],
                                train_data["cpt_ids"][positions],
                                train_data["icd_ids"][positions],
                                train_data["ttnc"][positions],
                                train_data["previous_cpt_ids"][positions],
                                train_data["previous_icd_ids"][positions],
                                train_data["previous_ttnc"][positions],
                                cpt_vocab_size=config.cpt_vocab_size,
                                icd_vocab_size=config.icd_vocab_size,
                                ttnc_vocab_size=config.ttnc_vocab_size,
                                max_cpt_tokens=config.max_cpt_tokens,
                                max_icd_tokens=config.max_icd_tokens,
                                hidden_dim=args.hidden_dim,
                                epochs=args.epochs,
                                batch_size=args.decoder_batch_size,
                                lr=args.lr,
                                weight_decay=args.weight_decay,
                                event_ranking_weight=args.event_ranking_weight,
                                ttnc_copy_boost=args.copy_ttnc_logit_boost,
                                seed=seed,
                                device=device,
                            )
                        )
                        prediction = predict_add_remove_decoder(
                            decoder,
                            val_features,
                            feature_mean,
                            feature_std,
                            device,
                            previous_cpt_ids=val_data["previous_cpt_ids"],
                            previous_icd_ids=val_data["previous_icd_ids"],
                            previous_ttnc=val_data["previous_ttnc"],
                            max_cpt_tokens=config.max_cpt_tokens,
                            max_icd_tokens=config.max_icd_tokens,
                            ttnc_copy_boost=args.copy_ttnc_logit_boost,
                        )
                    else:
                        decoder, feature_mean, feature_std, history = fit_decoder(
                            train_features[positions],
                            train_data["cpt_ids"][positions],
                            train_data["icd_ids"][positions],
                            train_data["ttnc"][positions],
                            train_data["previous_cpt_ids"][positions],
                            train_data["previous_icd_ids"][positions],
                            train_data["previous_ttnc"][positions],
                            cpt_vocab_size=config.cpt_vocab_size,
                            icd_vocab_size=config.icd_vocab_size,
                            ttnc_vocab_size=config.ttnc_vocab_size,
                            max_cpt_tokens=config.max_cpt_tokens,
                            max_icd_tokens=config.max_icd_tokens,
                            hidden_dim=args.hidden_dim,
                            epochs=args.epochs,
                            batch_size=args.decoder_batch_size,
                            lr=args.lr,
                            weight_decay=args.weight_decay,
                            count_loss_weight=args.count_loss_weight,
                            copy_residual=args.copy_residual,
                            copy_logit_boost=args.copy_logit_boost,
                            copy_ttnc_logit_boost=args.copy_ttnc_logit_boost,
                            seed=seed,
                            device=device,
                        )
                        prediction = predict_decoder(
                            decoder,
                            val_features,
                            feature_mean,
                            feature_std,
                            device,
                            previous_cpt_ids=val_data["previous_cpt_ids"],
                            previous_icd_ids=val_data["previous_icd_ids"],
                            previous_ttnc=val_data["previous_ttnc"],
                            copy_residual=args.copy_residual,
                            copy_logit_boost=args.copy_logit_boost,
                            copy_ttnc_logit_boost=args.copy_ttnc_logit_boost,
                        )
                    generation_metrics = score_generation(
                        prediction,
                        val_cpt,
                        val_icd,
                        val_data["ttnc"],
                        train_cpt_support=train_cpt[positions].sum(axis=0),
                        train_icd_support=train_icd[positions].sum(axis=0),
                        max_cpt_tokens=config.max_cpt_tokens,
                        max_icd_tokens=config.max_icd_tokens,
                        min_class_support=args.min_class_support,
                    )
                    if args.add_remove:
                        generation_metrics["changes"] = score_change_events(
                            prediction,
                            val_cpt,
                            val_icd,
                            token_ids_to_multi_hot(
                                val_data["previous_cpt_ids"], config.cpt_vocab_size
                            ),
                            token_ids_to_multi_hot(
                                val_data["previous_icd_ids"], config.icd_vocab_size
                            ),
                        )
                    cost_metrics = compute_heldout_regression_probe_metrics(
                        train_features[positions],
                        train_data["cost"][positions],
                        val_features,
                        val_data["cost"],
                    )
                    run = {
                        "source": source,
                        "condition": condition,
                        "label_fraction": float(fraction),
                        "num_labels": int(len(positions)),
                        "seed": int(seed),
                        "copy_residual": bool(args.copy_residual),
                        "add_remove": bool(args.add_remove),
                        "event_ranking_weight": float(args.event_ranking_weight),
                        "generation": generation_metrics,
                        "cost": cost_metrics,
                        "training_loss_first": history[0],
                        "training_loss_last": history[-1],
                    }
                    runs.append(run)
                    stem = f"{source}__{condition}__f{fraction:g}__seed{seed}"
                    write_json(output_dir / f"{stem}.json", run)
                    torch.save(
                        {
                            "decoder_state_dict": decoder.state_dict(),
                            "feature_mean": feature_mean,
                            "feature_std": feature_std,
                            "source": source,
                            "condition": condition,
                            "label_fraction": float(fraction),
                            "seed": int(seed),
                        },
                        artifact_dir / f"{stem}.pt",
                    )
                    del decoder

    summary = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_contract": str(Path(args.data_contract).resolve()),
            "data_contract_hash": config.data_contract_hash,
            "vocab_hash": config.vocab_hash,
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "train_samples": int(len(train_data["cost"])),
            "validation_samples": int(len(val_data["cost"])),
            "sources": args.sources,
            "conditions": args.conditions,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "label_fractions": args.label_fractions,
            "copy_residual": bool(args.copy_residual),
            "add_remove": bool(args.add_remove),
            "event_ranking_weight": float(args.event_ranking_weight),
            "copy_logit_boost": float(args.copy_logit_boost),
            "copy_ttnc_logit_boost": float(args.copy_ttnc_logit_boost),
        },
        "empirical_prior": prior_metrics,
        "runs": runs,
        "aggregate": aggregate_runs(runs),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
