import argparse
import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.config import Config, apply_runtime_config_overrides, apply_training_recipe
from jepa_utils.data_prep import prepare_data


def build_synthetic_dataframe(num_patients: int, claims_per_patient: int) -> pd.DataFrame:
    rows = []
    for patient_idx in range(num_patients):
        tokens = []
        for claim_idx in range(claims_per_patient):
            tokens.extend(
                [
                    f"ttnc_specialty_{(patient_idx + claim_idx) % 5}",
                    f"cpt_proc_{(patient_idx * 3 + claim_idx) % 17}",
                    f"cpt_proc_{(patient_idx * 5 + claim_idx + 1) % 19}",
                    f"icd_diag_{(patient_idx * 7 + claim_idx) % 23}",
                    f"icd_diag_{(patient_idx * 11 + claim_idx + 2) % 29}",
                ]
            )

        rows.append(
            {
                "input": " ".join(tokens),
                "target": float(1000 + patient_idx * 125),
            }
        )

    return pd.DataFrame(rows)


def build_smoke_config(data_path: str, recipe_name: str, recipe_overrides: dict | None = None) -> Config:
    config = apply_training_recipe(Config(), recipe_name)
    config.data_path = data_path
    config.embedding_dim = 32
    config.hidden_dim = 64
    config.rnn_hidden_dim = 64
    config.output_dim = 32
    config.ff_hidden_dim = 64
    config.sae_hidden_dim = 64
    config.sae_k = 4
    config.max_claims_len = 8
    config.max_cpt_tokens = 3
    config.max_icd_tokens = 3
    config.use_plotting = False
    config.use_lr_find = False
    config.use_predictor_head = False
    config.use_token_prediction_head = False
    config.use_diffusion = False
    config.use_generative_save = False
    config.pretrain_diffusion = False
    config.representation_pretrain_epochs = 0
    config.generator_train_epochs = 0
    config.joint_train_epochs = 0
    config.current_stage = "smoke"

    for key, value in (recipe_overrides or {}).items():
        setattr(config, key, value)

    return apply_runtime_config_overrides(config)


def resolve_accelerator(accelerator: str) -> str:
    if accelerator != "auto":
        return accelerator
    return "gpu" if torch.cuda.is_available() else "cpu"


def format_prediction_shape(prediction: torch.Tensor) -> list[int]:
    return list(prediction.shape)


def run_recipe(
    name: str,
    recipe_overrides: dict,
    data_path: str,
    accelerator: str,
    log_dir: str,
) -> dict:
    pl.seed_everything(7, workers=True)
    config = build_smoke_config(data_path, name, recipe_overrides)
    _, train_dataloader, _, _, config, _ = prepare_data(config)

    model = HierarchicalClaimsModel(config)
    trainer = pl.Trainer(
        accelerator=resolve_accelerator(accelerator),
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=pl.loggers.CSVLogger(save_dir=log_dir, name=name),
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=1,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)

    batch = next(iter(train_dataloader))
    batch = tuple(t.to(model.device) for t in batch)
    model.eval()
    with torch.no_grad():
        outputs = model(
            cpt_tensor=batch[0],
            icd_tensor=batch[1],
            ttnc_tensor=batch[2],
            target=batch[3],
            teacher_forcing=True,
            generation=False,
        )

    result = {
        "recipe": name,
        "ssl_objective_type": config.ssl_objective_type,
        "target_encoder_mode": config.target_encoder_mode,
        "use_level2_dense_prediction": config.use_level2_dense_prediction,
        "observed_claim_k": int(config.observed_claim_k),
        "dataset_size": int(len(train_dataloader.dataset)),
        "train_batches": int(len(train_dataloader)),
        "loss": float(outputs["loss"].detach().cpu()),
        "ssl_loss_lvl2": float(outputs["ssl_loss_lvl2"].detach().cpu()),
        "ssl_predictive_lvl2": float(outputs["ssl_predictive_lvl2"].detach().cpu()),
        "ssl_regularizer_lvl2": float(outputs["ssl_regularizer_lvl2"].detach().cpu()),
        "prediction_shape": format_prediction_shape(outputs["prediction_lvl2"]),
        "clean_ssl_mode": bool(config.clean_ssl_mode),
    }

    if config.use_level2_dense_prediction:
        result["dense_observed_loss"] = float(outputs["dense_observed_loss"].detach().cpu())
        result["dense_next_loss"] = float(outputs["dense_next_loss"].detach().cpu())
    if config.use_levjepa_patient_views:
        result["levjepa_invariance_loss"] = float(
            outputs["levjepa_invariance_loss"].detach().cpu()
        )
        result["levjepa_sigreg_raw"] = float(
            outputs["levjepa_sigreg_raw"].detach().cpu()
        )
        result["levjepa_retained_claim_fraction"] = float(
            outputs["levjepa_retained_claim_fraction"].detach().cpu()
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one-batch Lightning smoke checks for JEPA SSL migration recipes."
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Trainer accelerator. Defaults to auto-detecting GPU when available.",
    )
    parser.add_argument(
        "--num-patients",
        type=int,
        default=24,
        help="Number of synthetic patients to write into the temporary parquet.",
    )
    parser.add_argument(
        "--claims-per-patient",
        type=int,
        default=4,
        help="Number of synthetic claims per patient. Must be >= 3 for dense smoke runs.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional path to save the recipe summaries as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.claims_per_patient < 3:
        raise ValueError("--claims-per-patient must be at least 3.")

    recipes = [
        (
            "vicreg_baseline",
            {},
        ),
        (
            "sigreg_core",
            {},
        ),
        (
            "sigreg_dense",
            {},
        ),
        (
            "levjepa_patient_views",
            {
                "sigreg_num_slices": 32,
                "levjepa_projector_hidden_dim": 64,
                "levjepa_projector_output_dim": 32,
            },
        ),
    ]

    with tempfile.TemporaryDirectory() as tmp_dir:
        data_path = Path(tmp_dir) / "synthetic_claims.parquet"
        build_synthetic_dataframe(
            num_patients=args.num_patients,
            claims_per_patient=args.claims_per_patient,
        ).to_parquet(data_path, index=False)

        results = []
        for name, overrides in recipes:
            result = run_recipe(
                name=name,
                recipe_overrides=overrides,
                data_path=str(data_path),
                accelerator=args.accelerator,
                log_dir=tmp_dir,
            )
            results.append(result)
            print(json.dumps(result, indent=2))

        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
