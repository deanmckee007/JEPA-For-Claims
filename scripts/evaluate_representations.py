import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.config import (
    Config,
    apply_config_overrides,
    apply_runtime_config_overrides,
    apply_training_recipe,
    get_training_recipe_names,
)
from jepa_utils.data_prep import prepare_data
from jepa_utils.checkpointing import load_claims_model_checkpoint
from jepa_utils.representation_eval import (
    collect_patient_representations,
    compute_attentive_regression_probe_metrics,
    compute_claim_prototype_metrics,
    compute_claim_prototype_stability,
    compute_heldout_regression_probe_metrics,
    compute_missing_modality_metrics,
    evaluate_representation_quality,
    extract_raw_sequence_lengths,
    get_representation_source_names,
    save_representation_eval,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate JEPA patient representations on simple downstream probes."
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint to load.")
    parser.add_argument("--data-path", type=str, default=None, help="Path to the parquet dataset.")
    parser.add_argument("--data-contract", type=str, default=None, help="Frozen split/vocabulary contract.")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true")
    parser.add_argument(
        "--recipe",
        choices=get_training_recipe_names(),
        default="custom",
        help="Recipe preset used to reconstruct the model config.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Device selection for embedding extraction.",
    )
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for embedding extraction.")
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum number of samples to score.")
    parser.add_argument("--retrieval-k", type=int, default=5, help="Neighborhood size for retrieval metrics.")
    parser.add_argument(
        "--attentive-probe",
        action="store_true",
        help="Fit a frozen query-attention probe over per-claim sequence states.",
    )
    parser.add_argument("--attentive-probe-epochs", type=int, default=20)
    parser.add_argument("--attentive-probe-batch-size", type=int, default=256)
    parser.add_argument("--attentive-probe-lr", type=float, default=1e-3)
    parser.add_argument("--attentive-probe-weight-decay", type=float, default=1e-4)
    parser.add_argument("--attentive-probe-num-heads", type=int, default=4)
    parser.add_argument(
        "--attentive-probe-train-max-samples",
        type=int,
        default=10000,
        help="Maximum frozen training sequences retained for the attentive probe.",
    )
    parser.add_argument(
        "--representation-source",
        choices=get_representation_source_names(),
        default="patient_representation_pre_sae",
        help="Which sequence-level representation to feed into downstream probes.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for evaluation and downstream probes.")
    parser.add_argument(
        "--output-json",
        type=str,
        default="representation_eval.json",
        help="Path to write the evaluation report.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        help="Config override in key=value form. Repeat to set multiple fields.",
    )
    return parser.parse_args(argv)


def resolve_device(accelerator: str) -> torch.device:
    if accelerator == "cpu":
        return torch.device("cpu")
    if accelerator == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU requested for evaluation, but CUDA is not available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_eval_config(args):
    config = apply_training_recipe(Config(), args.recipe)
    if args.data_path:
        config.data_path = args.data_path
    if args.data_contract:
        config.data_contract_path = args.data_contract
    config.evaluation_split = args.split
    config.allow_legacy_checkpoint_loading = args.allow_legacy_checkpoint
    if args.seed is not None:
        config.seed = args.seed

    config.use_plotting = False
    config.use_generative_save = False
    config.pretrain_diffusion = False
    config.trainer_accelerator = args.accelerator

    config = apply_config_overrides(config, getattr(args, "config_overrides", None))
    return apply_runtime_config_overrides(config)


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.accelerator)
    config = build_eval_config(args)
    pl.seed_everything(config.seed, workers=True)

    train_dataset, _, eval_dataset, _, config, dataset = prepare_data(
        config,
        requested_eval_split=args.split,
    )
    train_probe_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_eval_fn,
        shuffle=False,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_eval_fn,
        shuffle=False,
    )

    model = load_claims_model_checkpoint(
        HierarchicalClaimsModel,
        args.checkpoint,
        config=config,
        map_location=device,
        allow_legacy=config.allow_legacy_checkpoint_loading,
    )

    train_embeddings, _, train_targets, train_metadata = collect_patient_representations(
        model,
        train_probe_dataloader,
        device=device,
        max_samples=None,
        representation_source=args.representation_source,
        include_sequence_states=args.attentive_probe,
        sequence_state_max_samples=args.attentive_probe_train_max_samples,
    )
    embeddings, specialty_labels, regression_targets, metadata = collect_patient_representations(
        model,
        eval_dataloader,
        device=device,
        max_samples=args.max_samples,
        representation_source=args.representation_source,
        include_sequence_states=args.attentive_probe,
    )
    raw_sequence_lengths = extract_raw_sequence_lengths(eval_dataset, max_samples=args.max_samples)
    effective_sequence_lengths = metadata.get("effective_sequence_lengths", metadata.get("sequence_lengths"))
    results = evaluate_representation_quality(
        embeddings,
        specialty_labels,
        regression_targets,
        sequence_lengths=raw_sequence_lengths,
        effective_sequence_lengths=effective_sequence_lengths,
        retrieval_k=args.retrieval_k,
        random_state=config.seed,
    )
    results.update(
        compute_heldout_regression_probe_metrics(
            train_embeddings,
            train_targets,
            embeddings,
            regression_targets,
        )
    )
    results["target_probe_protocol"] = "frozen_train_fit_to_heldout_eval"
    if args.attentive_probe:
        attentive_train_samples = train_metadata["sequence_states"].shape[0]
        results["attentive_probe"] = compute_attentive_regression_probe_metrics(
            train_metadata["sequence_states"],
            train_metadata["sequence_state_masks"],
            train_targets[:attentive_train_samples],
            metadata["sequence_states"],
            metadata["sequence_state_masks"],
            regression_targets,
            device=device,
            epochs=args.attentive_probe_epochs,
            batch_size=args.attentive_probe_batch_size,
            learning_rate=args.attentive_probe_lr,
            weight_decay=args.attentive_probe_weight_decay,
            num_heads=args.attentive_probe_num_heads,
            random_state=config.seed,
        )
        results["attentive_probe_protocol"] = (
            "frozen_claim_sequence_query_attention_train_fit_to_heldout_eval"
        )
    if "claim_prototype_assignments" in metadata:
        results["claim_prototypes"] = compute_claim_prototype_metrics(
            metadata["claim_prototype_assignments"],
            metadata["claim_prototype_probabilities"],
        )
    if getattr(model, "use_composable_level1", False):
        missing_modality_results = {}
        missing_prototype_results = {}
        for modality in ("cpt", "icd"):
            missing_embeddings, _, missing_targets, missing_metadata = collect_patient_representations(
                model,
                eval_dataloader,
                device=device,
                max_samples=args.max_samples,
                representation_source=args.representation_source,
                missing_modality=modality,
            )
            missing_modality_results[f"missing_{modality}"] = (
                compute_missing_modality_metrics(
                    train_embeddings,
                    train_targets,
                    embeddings,
                    missing_embeddings,
                    missing_targets,
                )
            )
            if (
                "claim_prototype_probabilities" in metadata
                and "claim_prototype_probabilities" in missing_metadata
            ):
                missing_prototype_results[f"missing_{modality}"] = (
                    compute_claim_prototype_stability(
                        metadata["claim_prototype_probabilities"],
                        missing_metadata["claim_prototype_probabilities"],
                    )
                )
        results["missing_modality"] = missing_modality_results
        if missing_prototype_results:
            results["claim_prototype_missing_modality_stability"] = (
                missing_prototype_results
            )
    results.update(
        {
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "data_path": str(Path(config.data_path).resolve()),
            "recipe": config.train_recipe,
            "device": str(device),
            "seed": int(config.seed),
            "representation_source": args.representation_source,
            "evaluation_weights": metadata.get("evaluation_weights", "online"),
            "evaluation_split": args.split,
            "data_contract_hash": config.data_contract_hash,
            "vocab_hash": config.vocab_hash,
            "representation_sharing": model.representation_sharing_contract(),
            "sigreg_formulation": getattr(config, "sigreg_formulation", None),
            "sigreg_weight_lvl1": float(config.sigreg_weight_lvl1),
            "sigreg_weight_lvl2": float(config.sigreg_weight_lvl2),
            "claim_prototype_weight": float(
                getattr(config, "claim_prototype_weight", 0.0)
                if getattr(config, "use_claim_prototypes", False)
                else 0.0
            ),
        }
    )

    save_representation_eval(results, args.output_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
