"""Test whether next-claim supervision improves a low-label frozen cost probe."""

import argparse
import copy
import json
import math
import sys
from collections import defaultdict
from itertools import cycle
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
from jepa_models.ssl_objectives import sigreg_gaussian_distance
from jepa_utils.checkpointing import load_claims_model_checkpoint, read_checkpoint_config
from jepa_utils.config import apply_runtime_config_overrides
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import score_regression_predictions
from scripts.run_cost_supervision_ablation import stratified_fraction_positions
from scripts.run_frozen_generation_probe import (
    balanced_multilabel_loss,
    collect_frozen_dataset,
    resolve_device,
    token_ids_to_multi_hot,
    write_json,
)


class ResidualFeatureAdapter(nn.Module):
    """Zero-initialized bottleneck that starts as an exact identity map."""

    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.down = nn.Linear(input_dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, input_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, features):
        return features + self.up(F.gelu(self.down(self.norm(features))))


class SharedCostGenerationHead(nn.Module):
    def __init__(
        self, input_dim, hidden_dim, cpt_vocab_size, icd_vocab_size,
        ttnc_vocab_size, adapter_dim=0, linear_heads=False,
    ):
        super().__init__()
        self.adapter = (
            ResidualFeatureAdapter(input_dim, adapter_dim) if adapter_dim > 0 else nn.Identity()
        )
        if linear_heads:
            self.trunk = nn.Identity()
            output_dim = input_dim
        else:
            self.trunk = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            output_dim = hidden_dim
        self.cost_head = nn.Linear(output_dim, 1)
        self.cpt_head = nn.Linear(output_dim, cpt_vocab_size)
        self.icd_head = nn.Linear(output_dim, icd_vocab_size)
        self.ttnc_head = nn.Linear(output_dim, ttnc_vocab_size)

    def adapted_features(self, features):
        return self.adapter(features)

    def shared_features(self, features):
        return self.trunk(self.adapted_features(features))

    def forward(self, features):
        hidden = self.shared_features(features)
        return {
            "cost": self.cost_head(hidden).squeeze(1),
            "cpt_logits": self.cpt_head(hidden),
            "icd_logits": self.icd_head(hidden),
            "ttnc_logits": self.ttnc_head(hidden),
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["auto", "cpu", "gpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--head-batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--adapter-dim", type=int, default=0)
    parser.add_argument("--adapter-l2-weight", type=float, default=0.0)
    parser.add_argument("--linear-heads", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cost-label-fraction", type=float, default=0.1)
    parser.add_argument("--aux-weights", nargs="+", type=float, default=[0.0, 0.03, 0.1])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--conditions", nargs="+", choices=["pretrained", "random_encoder"],
        default=["pretrained", "random_encoder"],
    )
    parser.add_argument("--random-encoder-seed", type=int, default=314159)
    args = parser.parse_args(argv)
    if not 0 < args.cost_label_fraction <= 1:
        parser.error("--cost-label-fraction must be in (0, 1]")
    if any(weight < 0 for weight in args.aux_weights):
        parser.error("--aux-weights must be nonnegative")
    return args


def initialize_generation_priors(model, cpt, icd, ttnc):
    with torch.no_grad():
        for head, target in ((model.cpt_head, cpt), (model.icd_head, icd)):
            prevalence = np.clip(target.mean(axis=0), 1e-7, 1 - 1e-7)
            head.weight.zero_()
            head.bias.copy_(torch.from_numpy(np.log(prevalence / (1 - prevalence))))
        counts = np.bincount(ttnc, minlength=model.ttnc_head.out_features).astype(np.float32)
        counts[0] = 0
        probability = (counts + 1e-3) / (counts.sum() + 1e-3 * len(counts))
        model.ttnc_head.weight.zero_()
        model.ttnc_head.bias.copy_(torch.from_numpy(np.log(np.clip(probability, 1e-12, 1))))
        model.cost_head.weight.zero_()
        model.cost_head.bias.zero_()


def auxiliary_loss(outputs, cpt, icd, ttnc, modalities=("cpt", "icd", "ttnc")):
    ttnc_logits = outputs["ttnc_logits"].clone()
    ttnc_logits[:, 0] = torch.finfo(ttnc_logits.dtype).min
    components = {
        "cpt": (0.5, balanced_multilabel_loss(outputs["cpt_logits"][:, 1:], cpt[:, 1:])),
        "icd": (0.5, balanced_multilabel_loss(outputs["icd_logits"][:, 1:], icd[:, 1:])),
        "ttnc": (1.0, F.cross_entropy(ttnc_logits, ttnc)),
    }
    unknown = set(modalities).difference(components)
    if unknown:
        raise ValueError(f"Unsupported auxiliary modalities: {sorted(unknown)}")
    if not modalities:
        raise ValueError("At least one auxiliary modality is required")
    nominal_weight = sum(components[name][0] for name in modalities)
    return 2.0 * sum(
        components[name][0] * components[name][1] for name in modalities
    ) / nominal_weight


def fit_head(
    train_features,
    train_cost,
    train_cpt_ids,
    train_icd_ids,
    train_ttnc,
    labeled_positions,
    *,
    cpt_vocab_size,
    icd_vocab_size,
    ttnc_vocab_size,
    hidden_dim,
    batch_size,
    epochs,
    lr,
    weight_decay,
    adapter_dim,
    adapter_l2_weight,
    linear_heads,
    aux_weight,
    seed,
    device,
    aux_modalities=("cpt", "icd", "ttnc"),
    aux_label_mode="aligned",
    consistency_weight=0.0,
    consistency_noise_std=0.1,
    sigreg_weight=0.0,
    sigreg_num_slices=256,
    sigreg_num_points=17,
    sigreg_formulation="lejepa_convex",
):
    if aux_weight > 0 and consistency_weight > 0:
        raise ValueError("Supervised auxiliary and consistency losses are mutually exclusive")
    pl.seed_everything(seed, workers=True)
    mean = train_features.mean(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(train_features.std(axis=0, keepdims=True), 1e-6).astype(np.float32)
    features = ((train_features - mean) / std).astype(np.float32)
    cpt = token_ids_to_multi_hot(train_cpt_ids, cpt_vocab_size)
    icd = token_ids_to_multi_hot(train_icd_ids, icd_vocab_size)
    ttnc = np.asarray(train_ttnc, dtype=np.int64)
    if aux_label_mode == "shuffled":
        permutation = np.random.default_rng(seed + 4001).permutation(len(cpt))
        cpt = cpt[permutation]
        icd = icd[permutation]
        ttnc = ttnc[permutation]
    elif aux_label_mode != "aligned":
        raise ValueError("aux_label_mode must be 'aligned' or 'shuffled'")
    cost_mean = float(np.mean(train_cost[labeled_positions]))
    cost_std = float(max(np.std(train_cost[labeled_positions]), 1e-6))
    normalized_cost = ((train_cost - cost_mean) / cost_std).astype(np.float32)

    generation_dataset = TensorDataset(
        torch.from_numpy(features), torch.from_numpy(cpt), torch.from_numpy(icd),
        torch.from_numpy(ttnc),
    )
    cost_dataset = TensorDataset(
        torch.from_numpy(features[labeled_positions]),
        torch.from_numpy(normalized_cost[labeled_positions]),
    )
    generation_loader = DataLoader(
        generation_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 11),
    )
    cost_loader = DataLoader(
        cost_dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed + 23),
    )
    model = SharedCostGenerationHead(
        features.shape[1], hidden_dim, cpt_vocab_size, icd_vocab_size, ttnc_vocab_size,
        adapter_dim=adapter_dim, linear_heads=linear_heads,
    ).to(device)
    initialize_generation_priors(model, cpt, icd, ttnc)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    history = []
    for _ in range(epochs):
        model.train()
        losses = []
        cost_batches = cycle(cost_loader)
        for gen_features, gen_cpt, gen_icd, gen_ttnc in generation_loader:
            cost_features, cost_target = next(cost_batches)
            cost_features = cost_features.to(device)
            cost_target = cost_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            cost_output = model(cost_features)
            cost_loss = F.mse_loss(cost_output["cost"], cost_target)
            if aux_weight > 0:
                gen_features = gen_features.to(device)
                gen_output = model(gen_features)
                gen_loss = auxiliary_loss(
                    gen_output, gen_cpt.to(device), gen_icd.to(device), gen_ttnc.to(device),
                    modalities=aux_modalities,
                )
                loss = cost_loss + aux_weight * gen_loss
                penalty_features = gen_features
            elif consistency_weight > 0:
                gen_features = gen_features.to(device)
                clean_hidden = model.shared_features(gen_features)
                noisy_hidden = model.shared_features(
                    gen_features
                    + consistency_noise_std * torch.randn_like(gen_features)
                )
                gen_loss = F.mse_loss(noisy_hidden, clean_hidden.detach())
                loss = cost_loss + consistency_weight * gen_loss
                penalty_features = gen_features
            else:
                gen_loss = cost_loss.new_zeros(())
                loss = cost_loss
                penalty_features = cost_features
            if sigreg_weight > 0:
                gen_features = gen_features.to(device)
                sigreg_hidden = model.shared_features(gen_features)
                sigreg_loss = sigreg_gaussian_distance(
                    sigreg_hidden,
                    num_slices=sigreg_num_slices,
                    num_points=sigreg_num_points,
                    formulation=sigreg_formulation,
                )
                loss = loss + sigreg_weight * sigreg_loss
                penalty_features = gen_features
            else:
                sigreg_loss = cost_loss.new_zeros(())
            if adapter_l2_weight > 0:
                adapted = model.adapted_features(penalty_features)
                adapter_penalty = F.mse_loss(adapted, penalty_features)
                loss = loss + adapter_l2_weight * adapter_penalty
            loss.backward()
            optimizer.step()
            losses.append((
                float(cost_loss.detach().cpu()),
                float(gen_loss.detach().cpu()),
                float(sigreg_loss.detach().cpu()),
            ))
        history.append({
            "cost": float(np.mean([row[0] for row in losses])),
            "generation": float(np.mean([row[1] for row in losses])),
            "sigreg": float(np.mean([row[2] for row in losses])),
        })
    return model, mean, std, cost_mean, cost_std, history


def evaluate_head(model, features, mean, std, cost_mean, cost_std, targets, device):
    normalized = ((features - mean) / std).astype(np.float32)
    collected = defaultdict(list)
    geometry = defaultdict(list)
    hidden_chunks = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(normalized), 1024):
            batch = torch.from_numpy(normalized[start:start + 1024]).to(device)
            adapted = model.adapted_features(batch)
            shared_hidden = model.shared_features(batch)
            outputs = model(batch)
            for key, value in outputs.items():
                collected[key].append(value.cpu())
            displacement = adapted - batch
            geometry["relative_l2"].append(
                (displacement.norm(dim=1) / batch.norm(dim=1).clamp_min(1e-8)).cpu()
            )
            geometry["cosine"].append(F.cosine_similarity(adapted, batch, dim=1).cpu())
            hidden_chunks.append(shared_hidden.cpu())
    outputs = {key: torch.cat(value).numpy() for key, value in collected.items()}
    predicted_log_cost = outputs["cost"] * cost_std + cost_mean
    cost_metrics = score_regression_predictions(predicted_log_cost, targets["cost"])
    cpt = token_ids_to_multi_hot(targets["cpt_ids"], outputs["cpt_logits"].shape[1])
    icd = token_ids_to_multi_hot(targets["icd_ids"], outputs["icd_logits"].shape[1])
    cpt_probability = torch.sigmoid(torch.from_numpy(outputs["cpt_logits"])).numpy()
    icd_probability = torch.sigmoid(torch.from_numpy(outputs["icd_logits"])).numpy()
    ttnc_logits = outputs["ttnc_logits"].copy()
    ttnc_logits[:, 0] = -1e9
    generation_metrics = {
        "cpt_micro_average_precision": float(average_precision_score(cpt[:, 1:].ravel(), cpt_probability[:, 1:].ravel())),
        "icd_micro_average_precision": float(average_precision_score(icd[:, 1:].ravel(), icd_probability[:, 1:].ravel())),
        "ttnc_accuracy": float(np.mean(ttnc_logits.argmax(axis=1) == targets["ttnc"])),
    }
    adapter_geometry = {
        "mean_relative_l2": float(torch.cat(geometry["relative_l2"]).mean()),
        "mean_cosine_to_frozen": float(torch.cat(geometry["cosine"]).mean()),
    }
    hidden = torch.cat(hidden_chunks)
    adapter_geometry.update({
        "shared_hidden_mean_feature_std": float(hidden.std(dim=0).mean()),
        "shared_hidden_mean_norm": float(hidden.norm(dim=1).mean()),
    })
    return cost_metrics, generation_metrics, adapter_geometry


def aggregate_runs(runs):
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["condition"], run["aux_weight"])].append(run)
    result = {}
    metric_names = (
        "target_probe_mae_dollars", "target_probe_rmse_dollars",
        "target_probe_wape_percent", "target_probe_rmse_log1p",
    )
    for (condition, weight), rows in grouped.items():
        metrics = {}
        for name in metric_names:
            values = [row["cost"][name] for row in rows]
            metrics[name] = {
                "mean": float(np.mean(values)),
                "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            }
        result[f"{condition}|{weight:g}"] = {"num_runs": len(rows), "cost": metrics}
    return result


def paired_deltas(runs):
    by_key = {(row["condition"], row["aux_weight"], row["seed"]): row for row in runs}
    result = {}
    conditions = sorted({row["condition"] for row in runs})
    weights = sorted({row["aux_weight"] for row in runs if row["aux_weight"] > 0})
    seeds = sorted({row["seed"] for row in runs})
    for condition in conditions:
        for weight in weights:
            deltas = []
            for seed in seeds:
                baseline = by_key.get((condition, 0.0, seed))
                auxiliary = by_key.get((condition, weight, seed))
                if baseline and auxiliary:
                    deltas.append(auxiliary["cost"]["target_probe_mae_dollars"] - baseline["cost"]["target_probe_mae_dollars"])
            if deltas:
                result[f"{condition}|{weight:g}"] = {
                    "mae_dollars_aux_minus_cost_only_mean": float(np.mean(deltas)),
                    "sample_std": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
                    "paired_seeds": len(deltas),
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
    train_dataset, _, val_dataset, _, config, dataset = prepare_data(config, requested_eval_split="val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_eval_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_eval_fn)

    pretrained_model = load_claims_model_checkpoint(
        HierarchicalClaimsModel, args.checkpoint, config=config, map_location=device
    )
    feature_sources = ["next_claim_prediction"]
    pretrained_train = collect_frozen_dataset(pretrained_model, train_loader, feature_sources, device)
    pretrained_val = collect_frozen_dataset(pretrained_model, val_loader, feature_sources, device)
    del pretrained_model
    datasets = {"pretrained": (pretrained_train, pretrained_val)}
    if "random_encoder" in args.conditions:
        pl.seed_everything(args.random_encoder_seed, workers=True)
        random_model = HierarchicalClaimsModel(copy.deepcopy(config))
        random_train = collect_frozen_dataset(random_model, train_loader, feature_sources, device)
        random_val = collect_frozen_dataset(random_model, val_loader, feature_sources, device)
        del random_model
        datasets["random_encoder"] = (random_train, random_val)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    runs = []
    artifact_dir = output_dir / "heads"
    artifact_dir.mkdir(exist_ok=True)
    for condition in args.conditions:
        train_data, val_data = datasets[condition]
        train_features = train_data["features"]["next_claim_prediction"]
        val_features = val_data["features"]["next_claim_prediction"]
        for seed in args.seeds:
            positions = stratified_fraction_positions(train_data["cost"], args.cost_label_fraction, seed)
            for aux_weight in args.aux_weights:
                print(f"Training condition={condition} seed={seed} aux_weight={aux_weight:g} cost_labels={len(positions)}")
                model, mean, std, cost_mean, cost_std, history = fit_head(
                    train_features, train_data["cost"], train_data["cpt_ids"],
                    train_data["icd_ids"], train_data["ttnc"], positions,
                    cpt_vocab_size=config.cpt_vocab_size,
                    icd_vocab_size=config.icd_vocab_size,
                    ttnc_vocab_size=config.ttnc_vocab_size,
                    hidden_dim=args.hidden_dim, batch_size=args.head_batch_size,
                    epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                    adapter_dim=args.adapter_dim, adapter_l2_weight=args.adapter_l2_weight,
                    linear_heads=args.linear_heads,
                    aux_weight=aux_weight, seed=seed, device=device,
                )
                cost_metrics, generation_metrics, adapter_geometry = evaluate_head(
                    model, val_features, mean, std, cost_mean, cost_std, val_data, device
                )
                run = {
                    "condition": condition, "seed": int(seed),
                    "aux_weight": float(aux_weight), "cost_label_fraction": float(args.cost_label_fraction),
                    "num_cost_labels": int(len(positions)), "cost": cost_metrics,
                    "generation": generation_metrics, "training_first": history[0],
                    "training_last": history[-1], "adapter_geometry": adapter_geometry,
                }
                runs.append(run)
                stem = f"{condition}__seed{seed}__aux{aux_weight:g}"
                write_json(output_dir / f"{stem}.json", run)
                torch.save({"state_dict": model.state_dict(), "feature_mean": mean, "feature_std": std,
                            "cost_mean": cost_mean, "cost_std": cost_std}, artifact_dir / f"{stem}.pt")
                del model

    summary = {
        "protocol": {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_contract": str(Path(args.data_contract).resolve()),
            "data_contract_hash": config.data_contract_hash,
            "vocab_hash": config.vocab_hash,
            "representation_source": "next_claim_prediction",
            "encoder_frozen": True,
            "cost_label_fraction": args.cost_label_fraction,
            "generation_label_fraction": 1.0,
            "matched_update_count": True,
            "evaluation_split": "frozen_validation",
            "test_accessed": False,
            "seeds": args.seeds,
            "aux_weights": args.aux_weights,
            "adapter_dim": args.adapter_dim,
            "adapter_l2_weight": args.adapter_l2_weight,
            "linear_heads": args.linear_heads,
        },
        "runs": runs,
        "aggregate": aggregate_runs(runs),
        "paired_deltas": paired_deltas(runs),
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps({"aggregate": summary["aggregate"], "paired_deltas": summary["paired_deltas"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
