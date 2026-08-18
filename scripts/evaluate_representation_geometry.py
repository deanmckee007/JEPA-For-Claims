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
    compute_representation_geometry_metrics,
    get_representation_source_names,
    save_representation_eval,
    select_representation_tensor,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate representation-geometry diagnostics without downstream probes."
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
    parser.add_argument("--max-samples", type=int, default=20000, help="Maximum number of samples to score.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for evaluation.")
    parser.add_argument(
        "--representation-source",
        dest="representation_sources",
        action="append",
        choices=get_representation_source_names(),
        default=None,
        help="Representation source to score. Repeat to evaluate multiple sources.",
    )
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Evaluate every supported representation source.",
    )
    parser.add_argument(
        "--cosine-sample-size",
        type=int,
        default=2048,
        help="Number of embeddings used for pairwise cosine diagnostics.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="representation_geometry.json",
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


def collect_representation_geometry(
    model,
    dataloader,
    representation_sources,
    device,
    max_samples,
    cosine_sample_size,
):
    model = model.to(device)
    model.eval()

    embedding_chunks = {source: [] for source in representation_sources}

    with torch.no_grad():
        for batch in dataloader:
            cpt_tensor, icd_tensor, ttnc_tensor, target = batch
            outputs = model(
                cpt_tensor=cpt_tensor.to(device),
                icd_tensor=icd_tensor.to(device),
                ttnc_tensor=ttnc_tensor.to(device),
                target=target.to(device),
                teacher_forcing=True,
                generation=False,
            )

            for source in representation_sources:
                embedding_chunks[source].append(
                    select_representation_tensor(outputs, source).detach().cpu()
                )

            if max_samples is not None:
                current_size = sum(chunk.size(0) for chunk in embedding_chunks[representation_sources[0]])
                if current_size >= max_samples:
                    break

    metrics = {}
    for source, chunks in embedding_chunks.items():
        if not chunks:
            continue
        embeddings = torch.cat(chunks, dim=0).numpy()
        if max_samples is not None:
            embeddings = embeddings[:max_samples]
        metrics[source] = compute_representation_geometry_metrics(
            embeddings,
            cosine_sample_size=cosine_sample_size,
        )
    return metrics


def main(argv=None):
    args = parse_args(argv)
    device = resolve_device(args.accelerator)
    config = build_eval_config(args)
    pl.seed_everything(config.seed, workers=True)

    if args.all_sources:
        representation_sources = get_representation_source_names()
    else:
        representation_sources = args.representation_sources or ["patient_representation_pre_sae"]

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

    geometry = collect_representation_geometry(
        model,
        eval_dataloader,
        representation_sources=representation_sources,
        device=device,
        max_samples=args.max_samples,
        cosine_sample_size=args.cosine_sample_size,
    )

    results = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_path": str(Path(config.data_path).resolve()),
        "recipe": config.train_recipe,
        "device": str(device),
        "seed": int(config.seed),
        "representation_sources": representation_sources,
        "evaluation_split": args.split,
        "data_contract_hash": config.data_contract_hash,
        "vocab_hash": config.vocab_hash,
        "geometry": geometry,
    }
    save_representation_eval(results, args.output_json)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
