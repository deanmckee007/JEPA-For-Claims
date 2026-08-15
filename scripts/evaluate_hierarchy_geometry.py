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
from jepa_utils.representation_eval import compute_representation_geometry_metrics


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate participation-ratio geometry across the JEPA hierarchy."
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
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for extraction.")
    parser.add_argument("--max-samples", type=int, default=20000, help="Maximum number of patient samples.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for evaluation.")
    parser.add_argument(
        "--cosine-sample-size",
        type=int,
        default=2048,
        help="Number of embeddings used for pairwise cosine diagnostics.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="hierarchy_geometry.json",
        help="Path to write the geometry report.",
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


def _append_masked(chunks, key, tensor, mask=None):
    if tensor is None:
        return
    flat = tensor.reshape(-1, tensor.size(-1))
    if mask is not None:
        flat_mask = mask.reshape(-1).bool()
        flat = flat[flat_mask]
    if flat.numel() == 0:
        return
    chunks[key].append(flat.detach().cpu())


def collect_hierarchy_tensors(model, dataloader, device, max_samples=None):
    model = model.to(device)
    model.eval()

    hierarchy_chunks = {
        "context_claim_representation": [],
        "sequence_output": [],
        "context_mean_pool": [],
        "context_max_pool": [],
        "context_pooled": [],
        "predictive_state": [],
        "patient_representation_pre_sae": [],
        "patient_representation": [],
        "prediction_all_slots": [],
        "prediction_next_slot": [],
        "target_all_slots": [],
        "target_next_slot": [],
        "dense_decoder_latent": [],
    }
    if getattr(model, "use_level1", False):
        hierarchy_chunks.update(
            {
                "level1_context_cpt": [],
                "level1_context_icd": [],
                "level1_prediction_cpt": [],
                "level1_prediction_icd": [],
            }
        )

    num_patients = 0

    with torch.no_grad():
        for batch in dataloader:
            cpt_tensor, icd_tensor, ttnc_tensor, target = batch
            cpt_tensor = cpt_tensor.to(device)
            icd_tensor = icd_tensor.to(device)
            ttnc_tensor = ttnc_tensor.to(device)
            target = target.to(device)

            outputs = model(
                cpt_tensor=cpt_tensor,
                icd_tensor=icd_tensor,
                ttnc_tensor=ttnc_tensor,
                target=target,
                teacher_forcing=True,
                generation=False,
            )

            lvl2_targets = model._build_level2_ssl_targets(cpt_tensor, icd_tensor, ttnc_tensor)
            context_ttnc = lvl2_targets["context_ttnc"]
            context_claim_mask = context_ttnc != 0
            target_lvl2 = lvl2_targets["target_repr"]
            target_mask_lvl2 = lvl2_targets["target_mask"]

            context_lvl2 = model.encode_claims(
                lvl2_targets["context_cpt"],
                lvl2_targets["context_icd"],
                context_ttnc,
            )

            prediction_lvl2 = outputs["prediction_lvl2"]
            sequence_aux = outputs["sequence_aux"]

            _append_masked(
                hierarchy_chunks,
                "context_claim_representation",
                context_lvl2,
                mask=context_claim_mask,
            )
            _append_masked(
                hierarchy_chunks,
                "sequence_output",
                sequence_aux["sequence_output"],
                mask=context_claim_mask,
            )

            for key in (
                "context_mean_pool",
                "context_max_pool",
                "context_pooled",
                "predictive_state",
                "patient_representation_pre_sae",
                "patient_representation",
            ):
                _append_masked(hierarchy_chunks, key, outputs.get(key, sequence_aux.get(key)))

            dense_decoder_latent = sequence_aux.get("dense_decoder_latent")
            if dense_decoder_latent is not None:
                _append_masked(hierarchy_chunks, "dense_decoder_latent", dense_decoder_latent)

            _append_masked(
                hierarchy_chunks,
                "prediction_all_slots",
                prediction_lvl2,
                mask=target_mask_lvl2 if prediction_lvl2.dim() == 3 else None,
            )
            _append_masked(
                hierarchy_chunks,
                "target_all_slots",
                target_lvl2,
                mask=target_mask_lvl2 if target_lvl2.dim() == 3 else None,
            )

            prediction_next = prediction_lvl2[:, -1] if prediction_lvl2.dim() == 3 else prediction_lvl2
            target_next = target_lvl2[:, -1] if target_lvl2.dim() == 3 else target_lvl2
            next_mask = target_mask_lvl2[:, -1] if target_mask_lvl2.dim() == 2 else target_mask_lvl2
            _append_masked(hierarchy_chunks, "prediction_next_slot", prediction_next, mask=next_mask)
            _append_masked(hierarchy_chunks, "target_next_slot", target_next, mask=next_mask)

            if getattr(model, "use_level1", False):
                context_lvl1_cpt, mask_cpt = model.context_encoder_lvl1(cpt_tensor, "cpt")
                context_lvl1_icd, mask_icd = model.context_encoder_lvl1(icd_tensor, "icd")
                prediction_lvl1_cpt = model.prediction_block_lvl1(context_lvl1_cpt, mask_cpt)
                prediction_lvl1_icd = model.prediction_block_lvl1(context_lvl1_icd, mask_icd)
                _append_masked(hierarchy_chunks, "level1_context_cpt", context_lvl1_cpt, mask=mask_cpt)
                _append_masked(hierarchy_chunks, "level1_context_icd", context_lvl1_icd, mask=mask_icd)
                _append_masked(hierarchy_chunks, "level1_prediction_cpt", prediction_lvl1_cpt, mask=mask_cpt)
                _append_masked(hierarchy_chunks, "level1_prediction_icd", prediction_lvl1_icd, mask=mask_icd)

            num_patients += cpt_tensor.size(0)
            if max_samples is not None and num_patients >= max_samples:
                break

    hierarchy_arrays = {}
    for key, chunks in hierarchy_chunks.items():
        if not chunks:
            continue
        embeddings = torch.cat(chunks, dim=0)
        hierarchy_arrays[key] = embeddings.numpy()

    return hierarchy_arrays


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.accelerator)
    config = build_eval_config(args)
    pl.seed_everything(config.seed, workers=True)

    _, _, eval_dataset, _, config, dataset = prepare_data(
        config,
        requested_eval_split=args.split,
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

    hierarchy_arrays = collect_hierarchy_tensors(
        model,
        eval_dataloader,
        device=device,
        max_samples=args.max_samples,
    )
    geometry = {
        name: compute_representation_geometry_metrics(
            embeddings,
            cosine_sample_size=args.cosine_sample_size,
        )
        for name, embeddings in hierarchy_arrays.items()
    }

    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": str(Path(config.data_path).resolve()),
        "recipe": config.train_recipe,
        "device": str(device),
        "seed": int(config.seed),
        "geometry": geometry,
        "evaluation_split": args.split,
        "data_contract_hash": config.data_contract_hash,
        "vocab_hash": config.vocab_hash,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
