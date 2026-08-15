import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.config import Config, apply_runtime_config_overrides, apply_training_recipe
from jepa_utils.data_prep import prepare_data
from jepa_utils.representation_eval import (
    collect_patient_representations,
    compute_missing_modality_metrics,
)
from scripts.run_cost_supervision_ablation import (
    DirectCostHead,
    direct_cost_metrics,
    resolve_device,
    score_cost_head,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate a saved cost-supervision model/head under missing modalities."
    )
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--data-contract", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--accelerator", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-split-seed", type=int, default=42)
    parser.add_argument("--max-eval-pairs", type=int, default=500000)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args(argv)


def make_loader(subset, dataset, batch_size):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_eval_fn,
    )


def cosine_stability(full_embeddings, missing_embeddings):
    full_norm = np.linalg.norm(full_embeddings, axis=1)
    missing_norm = np.linalg.norm(missing_embeddings, axis=1)
    cosine = np.sum(full_embeddings * missing_embeddings, axis=1) / np.clip(
        full_norm * missing_norm,
        a_min=1e-8,
        a_max=None,
    )
    return {
        "full_to_missing_cosine_mean": float(cosine.mean()),
        "full_to_missing_cosine_median": float(np.median(cosine)),
    }


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.accelerator)
    config = apply_training_recipe(Config(), args.recipe)
    config.data_path = args.data_path
    config.data_contract_path = args.data_contract
    config.seed = args.seed
    config.data_split_seed = args.data_split_seed
    config.evaluation_split = "val"
    config.use_plotting = False
    config.use_generative_save = False
    config = apply_runtime_config_overrides(config)
    train_subset, _, val_subset, _, config, dataset = prepare_data(
        config,
        requested_eval_split="val",
    )
    train_loader = make_loader(train_subset, dataset, config.eval_batch_size)
    val_loader = make_loader(val_subset, dataset, config.eval_batch_size)

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    model = HierarchicalClaimsModel(config)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    head_state = artifact["head_state_dict"]
    hidden_dim, input_dim = head_state["network.0.weight"].shape
    head = DirectCostHead(input_dim, hidden_dim)
    head.load_state_dict(head_state, strict=True)
    model = model.to(device)
    head = head.to(device)

    train_embeddings, _, train_targets, _ = collect_patient_representations(
        model,
        train_loader,
        device=device,
        max_samples=None,
        representation_source="patient_representation_pre_sae",
    )
    full_embeddings, _, eval_targets, _ = collect_patient_representations(
        model,
        val_loader,
        device=device,
        max_samples=None,
        representation_source="patient_representation_pre_sae",
    )
    target_mean = float(artifact["target_mean"])
    target_std = float(artifact["target_std"])
    full_predictions = score_cost_head(
        head,
        full_embeddings,
        target_mean,
        target_std,
        device,
    )
    report = {
        "condition": artifact["condition"],
        "full": direct_cost_metrics(
            full_predictions,
            eval_targets,
            args.max_eval_pairs,
            args.seed,
        ),
        "missing_modality": {},
        "evaluation_split": "frozen_validation",
        "test_accessed": False,
        "artifact": str(Path(args.artifact).resolve()),
    }
    for modality in ("cpt", "icd"):
        missing_embeddings, _, missing_targets, _ = collect_patient_representations(
            model,
            val_loader,
            device=device,
            max_samples=None,
            representation_source="patient_representation_pre_sae",
            missing_modality=modality,
        )
        predictions = score_cost_head(
            head,
            missing_embeddings,
            target_mean,
            target_std,
            device,
        )
        direct = direct_cost_metrics(
            predictions,
            missing_targets,
            args.max_eval_pairs,
            args.seed,
        )
        direct.update(cosine_stability(full_embeddings, missing_embeddings))
        report["missing_modality"][f"missing_{modality}"] = {
            "direct_cost_head": direct,
            "frozen_linear_probe": compute_missing_modality_metrics(
                train_embeddings,
                train_targets,
                full_embeddings,
                missing_embeddings,
                missing_targets,
            ),
        }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
