import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Subset, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpointing import load_claims_model_checkpoint
from jepa_utils.config import (
    Config,
    apply_runtime_config_overrides,
    apply_training_recipe,
)
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import (
    collect_patient_representations,
    compute_heldout_regression_probe_metrics,
    compute_representation_geometry_metrics,
    compute_ttnc_proxy_clustering_metrics,
    cosine_retrieval_hit_rate_at_k,
    score_regression_predictions,
)
from scripts.evaluate_pairwise_cost_ranking import pairwise_accuracy, top_decile_recall


CONDITIONS = ("frozen_ssl", "ssl_finetune", "joint_ssl_cost", "supervised_scratch")


class DirectCostHead(nn.Module):
    """The same log-cost MLP used by every supervision condition."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, representations):
        return self.network(representations).squeeze(-1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare frozen SSL, SSL fine-tuning, joint SSL+cost, and "
            "supervised-only cost training on the frozen validation split."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--recipe", default="composable_level1_lejepa_any_code")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-split-seed",
        type=int,
        default=42,
        help="Frozen patient-split seed, independent of optimization seed.",
    )
    parser.add_argument("--head-epochs", type=int, default=30)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--joint-epochs", type=int, default=20)
    parser.add_argument("--scratch-epochs", type=int, default=20)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--finetune-encoder-lr", type=float, default=2e-5)
    parser.add_argument("--task-weight", type=float, default=1.0)
    parser.add_argument(
        "--end-to-end-label-fraction",
        type=float,
        default=1.0,
        help="Cost-stratified labeled fraction for fine-tune, joint, and scratch arms.",
    )
    parser.add_argument("--max-eval-pairs", type=int, default=500000)
    parser.add_argument(
        "--label-fractions",
        nargs="+",
        type=float,
        default=[0.01, 0.05, 0.10, 0.25, 1.0],
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=list(CONDITIONS),
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    if any(not 0.0 < fraction <= 1.0 for fraction in args.label_fractions):
        parser.error("label fractions must be in (0, 1]")
    if args.task_weight < 0:
        parser.error("task weight must be non-negative")
    if not 0.0 < args.end_to_end_label_fraction <= 1.0:
        parser.error("end-to-end label fraction must be in (0, 1]")
    return args


def resolve_device(accelerator):
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested but CUDA is unavailable")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stratified_fraction_positions(targets, fraction, seed, num_strata=10):
    """Select a deterministic cost-stratified fraction of training positions."""
    targets = np.asarray(targets)
    if fraction >= 1.0:
        return np.arange(len(targets), dtype=np.int64)
    sorted_positions = np.argsort(targets, kind="mergesort")
    strata = np.array_split(sorted_positions, min(num_strata, len(targets)))
    rng = np.random.default_rng(seed)
    selected = []
    for stratum in strata:
        shuffled = np.array(stratum, copy=True)
        rng.shuffle(shuffled)
        count = max(1, int(round(len(shuffled) * fraction)))
        selected.extend(shuffled[:count].tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def target_normalization(targets):
    targets = np.asarray(targets, dtype=np.float32)
    return float(targets.mean()), float(max(targets.std(), 1e-6))


def denormalize_cost_predictions(normalized_predictions, mean, std):
    return np.asarray(normalized_predictions) * std + mean


def combine_objectives(ssl_loss, task_loss, mode, task_weight):
    if mode == "joint_ssl_cost":
        return ssl_loss + task_weight * task_loss
    if mode in {"ssl_finetune", "supervised_scratch"}:
        return task_loss
    raise ValueError(f"Unsupported end-to-end mode: {mode}")


def make_loader(subset, dataset, batch_size, shuffle, seed):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate_fn if shuffle else dataset.collate_eval_fn,
        generator=torch.Generator().manual_seed(seed) if shuffle else None,
    )


def train_cost_head_on_embeddings(
    embeddings,
    targets,
    *,
    hidden_dim,
    epochs,
    lr,
    seed,
    device,
):
    pl.seed_everything(seed, workers=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)
    mean, std = target_normalization(targets)
    dataset = TensorDataset(
        torch.from_numpy(embeddings),
        torch.from_numpy((targets - mean) / std),
    )
    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    head = DirectCostHead(embeddings.shape[1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    epoch_losses = []
    for _ in range(epochs):
        head.train()
        losses = []
        for batch_embeddings, batch_targets in loader:
            batch_embeddings = batch_embeddings.to(device)
            batch_targets = batch_targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.mse_loss(head(batch_embeddings), batch_targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
    return head, mean, std, epoch_losses


def score_cost_head(head, embeddings, mean, std, device, batch_size=1024):
    head.eval()
    normalized = []
    with torch.no_grad():
        for start in range(0, len(embeddings), batch_size):
            batch = torch.from_numpy(
                np.asarray(embeddings[start : start + batch_size], dtype=np.float32)
            ).to(device)
            normalized.append(head(batch).cpu().numpy())
    return denormalize_cost_predictions(np.concatenate(normalized), mean, std)


def direct_cost_metrics(predictions, targets, max_eval_pairs, seed):
    metrics = score_regression_predictions(predictions, targets)
    accuracy, evaluated, available = pairwise_accuracy(
        predictions,
        targets,
        max_eval_pairs,
        seed,
    )
    metrics.update(
        {
            "pairwise_accuracy": accuracy,
            "pairwise_pairs_evaluated": evaluated,
            "pairwise_pairs_available": available,
            "spearman_correlation": float(spearmanr(predictions, targets).statistic),
            "top_decile_recall": top_decile_recall(predictions, targets),
        }
    )
    return metrics


def collect_split(model, loader, device):
    return collect_patient_representations(
        model,
        loader,
        device=device,
        max_samples=None,
        representation_source="patient_representation_pre_sae",
    )


def evaluate_model_and_head(
    model,
    head,
    train_loader,
    val_loader,
    *,
    target_mean,
    target_std,
    max_eval_pairs,
    seed,
    device,
):
    train_embeddings, _, train_targets, _ = collect_split(model, train_loader, device)
    val_embeddings, val_labels, val_targets, _ = collect_split(model, val_loader, device)
    predictions = score_cost_head(head, val_embeddings, target_mean, target_std, device)
    report = {
        "direct_cost_head": direct_cost_metrics(
            predictions,
            val_targets,
            max_eval_pairs,
            seed,
        ),
        "frozen_linear_probe": compute_heldout_regression_probe_metrics(
            train_embeddings,
            train_targets,
            val_embeddings,
            val_targets,
        ),
        "representation_geometry": compute_representation_geometry_metrics(val_embeddings),
        "ttnc_proxy_retrieval_hit_rate_at_5": cosine_retrieval_hit_rate_at_k(
            val_embeddings,
            val_labels,
            k=5,
        ),
    }
    report["representation_geometry"].update(
        compute_ttnc_proxy_clustering_metrics(val_embeddings, val_labels, random_state=seed)
    )
    return report


def train_end_to_end(
    model,
    train_loader,
    *,
    mode,
    epochs,
    input_dim,
    target_mean,
    target_std,
    hidden_dim,
    encoder_lr,
    head_lr,
    task_weight,
    device,
):
    model = model.to(device)
    head = DirectCostHead(input_dim, hidden_dim).to(device)
    model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": model_parameters, "lr": encoder_lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=1e-4,
    )
    epoch_losses = []
    epoch_task_losses = []
    epoch_ssl_losses = []
    for _ in range(epochs):
        model.train()
        head.train()
        losses = []
        task_losses = []
        ssl_losses = []
        for cpt, icd, ttnc, targets in train_loader:
            cpt = cpt.to(device)
            icd = icd.to(device)
            ttnc = ttnc.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(
                cpt,
                icd,
                ttnc,
                target=None,
                teacher_forcing=mode == "joint_ssl_cost",
                generation=False,
            )
            normalized_targets = (targets - target_mean) / target_std
            task_loss = nn.functional.mse_loss(
                head(outputs["patient_representation_pre_sae"]),
                normalized_targets,
            )
            ssl_loss = outputs["loss"]
            loss = combine_objectives(ssl_loss, task_loss, mode, task_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model_parameters) + list(head.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            task_losses.append(float(task_loss.detach().cpu()))
            ssl_losses.append(float(ssl_loss.detach().cpu()))
        epoch_losses.append(float(np.mean(losses)))
        epoch_task_losses.append(float(np.mean(task_losses)))
        epoch_ssl_losses.append(float(np.mean(ssl_losses)))
    return model, head, {
        "total": epoch_losses,
        "task": epoch_task_losses,
        "ssl_diagnostic": epoch_ssl_losses,
    }


def save_artifact(path, model, head, target_mean, target_std, condition, args):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "head_state_dict": head.cpu().state_dict(),
            "target_mean": target_mean,
            "target_std": target_std,
            "condition": condition,
            "args": vars(args),
        },
        path,
    )


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.accelerator)
    pl.seed_everything(args.seed, workers=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = apply_training_recipe(Config(), args.recipe)
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.seed = args.seed
    config.data_split_seed = args.data_split_seed
    config.evaluation_split = "val"
    config.use_plotting = False
    config.use_generative_save = False
    config.pretrain_diffusion = False
    config = apply_runtime_config_overrides(config)
    train_subset, _, val_subset, _, config, dataset = prepare_data(
        config,
        requested_eval_split="val",
    )
    config = apply_runtime_config_overrides(config)
    train_targets = np.asarray(
        [dataset.targets[train_subset.indices[position]] for position in range(len(train_subset))],
        dtype=np.float32,
    )
    target_mean, target_std = target_normalization(train_targets)
    train_loader = make_loader(
        train_subset,
        dataset,
        config.train_batch_size,
        True,
        args.seed,
    )
    train_eval_loader = make_loader(
        train_subset,
        dataset,
        config.eval_batch_size,
        False,
        args.seed,
    )
    val_loader = make_loader(
        val_subset,
        dataset,
        config.eval_batch_size,
        False,
        args.seed,
    )
    end_to_end_positions = stratified_fraction_positions(
        train_targets,
        args.end_to_end_label_fraction,
        args.seed,
    )
    end_to_end_subset = Subset(train_subset, end_to_end_positions.tolist())
    end_to_end_targets = train_targets[end_to_end_positions]
    end_to_end_target_mean, end_to_end_target_std = target_normalization(
        end_to_end_targets
    )
    end_to_end_train_loader = make_loader(
        end_to_end_subset,
        dataset,
        config.train_batch_size,
        True,
        args.seed,
    )
    end_to_end_train_eval_loader = make_loader(
        end_to_end_subset,
        dataset,
        config.eval_batch_size,
        False,
        args.seed,
    )

    base_model = None
    base_train_embeddings = None
    base_train_targets = None
    base_val_embeddings = None
    base_val_labels = None
    base_val_targets = None
    if "frozen_ssl" in args.conditions:
        base_model = load_claims_model_checkpoint(
            HierarchicalClaimsModel,
            args.checkpoint,
            config=config,
            map_location=device,
        )
        (
            base_train_embeddings,
            _,
            base_train_targets,
            _,
        ) = collect_split(base_model, train_eval_loader, device)
        (
            base_val_embeddings,
            base_val_labels,
            base_val_targets,
            _,
        ) = collect_split(base_model, val_loader, device)
        fraction_reports = {}
        full_head = None
        full_head_stats = None
        for fraction in sorted(set(args.label_fractions)):
            positions = stratified_fraction_positions(
                base_train_targets,
                fraction,
                args.seed,
            )
            head, mean, std, losses = train_cost_head_on_embeddings(
                base_train_embeddings[positions],
                base_train_targets[positions],
                hidden_dim=config.hidden_dim,
                epochs=args.head_epochs,
                lr=args.head_lr,
                seed=args.seed,
                device=device,
            )
            predictions = score_cost_head(
                head,
                base_val_embeddings,
                mean,
                std,
                device,
            )
            fraction_reports[str(fraction)] = {
                "num_labels": int(len(positions)),
                "metrics": direct_cost_metrics(
                    predictions,
                    base_val_targets,
                    args.max_eval_pairs,
                    args.seed,
                ),
                "training_loss_first": losses[0],
                "training_loss_last": losses[-1],
            }
            if math.isclose(fraction, 1.0):
                full_head = head
                full_head_stats = (mean, std)
        if full_head is None:
            positions = np.arange(len(base_train_targets))
            full_head, mean, std, _ = train_cost_head_on_embeddings(
                base_train_embeddings,
                base_train_targets,
                hidden_dim=config.hidden_dim,
                epochs=args.head_epochs,
                lr=args.head_lr,
                seed=args.seed,
                device=device,
            )
            full_head_stats = (mean, std)
        frozen_report = {
            "condition": "frozen_ssl",
            "protocol": "SSL encoder frozen; identical MLP fit on labeled train representations",
            "label_efficiency": fraction_reports,
            "full_representation_linear_probe": compute_heldout_regression_probe_metrics(
                base_train_embeddings,
                base_train_targets,
                base_val_embeddings,
                base_val_targets,
            ),
            "representation_geometry": compute_representation_geometry_metrics(
                base_val_embeddings
            ),
            "ttnc_proxy_retrieval_hit_rate_at_5": cosine_retrieval_hit_rate_at_k(
                base_val_embeddings,
                base_val_labels,
                k=5,
            ),
        }
        frozen_report["representation_geometry"].update(
            compute_ttnc_proxy_clustering_metrics(
                base_val_embeddings,
                base_val_labels,
                random_state=args.seed,
            )
        )
        write_json(output_dir / "frozen_ssl.json", frozen_report)
        save_artifact(
            output_dir / "frozen_ssl.pt",
            base_model,
            full_head,
            full_head_stats[0],
            full_head_stats[1],
            "frozen_ssl",
            args,
        )
        del base_model, full_head
        torch.cuda.empty_cache() if device.type == "cuda" else None

    epoch_map = {
        "ssl_finetune": args.finetune_epochs,
        "joint_ssl_cost": args.joint_epochs,
        "supervised_scratch": args.scratch_epochs,
    }
    for condition in args.conditions:
        if condition == "frozen_ssl":
            continue
        report_path = output_dir / f"{condition}.json"
        if args.skip_existing and report_path.exists():
            continue
        pl.seed_everything(args.seed, workers=True)
        if condition == "ssl_finetune":
            model = load_claims_model_checkpoint(
                HierarchicalClaimsModel,
                args.checkpoint,
                config=config,
                map_location=device,
            )
            encoder_lr = args.finetune_encoder_lr
        else:
            model = HierarchicalClaimsModel(copy.deepcopy(config))
            encoder_lr = args.encoder_lr
        model, head, loss_history = train_end_to_end(
            model,
            end_to_end_train_loader,
            mode=condition,
            epochs=epoch_map[condition],
            input_dim=config.patient_representation_dim,
            target_mean=end_to_end_target_mean,
            target_std=end_to_end_target_std,
            hidden_dim=config.hidden_dim,
            encoder_lr=encoder_lr,
            head_lr=args.head_lr,
            task_weight=args.task_weight,
            device=device,
        )
        report = evaluate_model_and_head(
            model,
            head,
            end_to_end_train_eval_loader,
            val_loader,
            target_mean=end_to_end_target_mean,
            target_std=end_to_end_target_std,
            max_eval_pairs=args.max_eval_pairs,
            seed=args.seed,
            device=device,
        )
        report.update(
            {
                "condition": condition,
                "epochs": epoch_map[condition],
                "task_weight": args.task_weight,
                "encoder_lr": encoder_lr,
                "head_lr": args.head_lr,
                "label_fraction": args.end_to_end_label_fraction,
                "num_labels": int(len(end_to_end_positions)),
                "loss_history": loss_history,
                "evaluation_split": "frozen_validation",
                "test_accessed": False,
            }
        )
        write_json(report_path, report)
        save_artifact(
            output_dir / f"{condition}.pt",
            model,
            head,
            end_to_end_target_mean,
            end_to_end_target_std,
            condition,
            args,
        )
        del model, head
        torch.cuda.empty_cache() if device.type == "cuda" else None

    summary = {}
    for condition in CONDITIONS:
        path = output_dir / f"{condition}.json"
        if path.exists():
            summary[condition] = json.loads(path.read_text(encoding="utf-8"))
    write_json(
        output_dir / "summary.json",
        {
            "protocol": {
                "selection_split": "frozen_validation",
                "test_accessed": False,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "data_contract": str(Path(args.data_contract).resolve()),
                "representation_source": "patient_representation_pre_sae",
                "target_normalization": "fixed_labeled_train_mean_and_std",
            },
            "conditions": summary,
        },
    )


if __name__ == "__main__":
    main()
