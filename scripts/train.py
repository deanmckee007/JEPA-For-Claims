# scripts/train.py
import argparse
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar, RichModelSummary
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jepa_utils.data_prep import prepare_data
from jepa_utils.checkpointing import load_claims_model_checkpoint
from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_models.diffusion import DiffusionModel
from jepa_utils.tensor_utils import calculate_entropy, adaptive_sampling
from jepa_utils.metrics import calculate_rmse
from jepa_utils.config import (
    Config,
    apply_config_overrides,
    apply_runtime_config_overrides,
    apply_training_recipe,
    get_training_recipe_names,
)
from jepa_utils.prediction_utils import decode_predicted_codes
import copy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train JEPA-for-Claims with optional SSL recipe presets."
    )
    parser.add_argument(
        "--recipe",
        choices=get_training_recipe_names(),
        default="custom",
        help="Named SSL recipe preset to apply before explicit CLI overrides.",
    )
    parser.add_argument("--data-path", type=str, default=None, help="Path to the parquet training dataset.")
    parser.add_argument("--data-contract", type=str, default=None, help="Frozen split/vocabulary contract.")
    parser.add_argument(
        "--create-data-contract",
        action="store_true",
        help="Create --data-contract if it does not exist; otherwise fail closed.",
    )
    parser.add_argument(
        "--accelerator",
        choices=["auto", "cpu", "gpu"],
        default=None,
        help="Lightning accelerator override.",
    )
    parser.add_argument("--devices", type=int, default=None, help="Lightning device count.")
    parser.add_argument("--representation-pretrain-epochs", type=int, default=None)
    parser.add_argument("--generator-train-epochs", type=int, default=None)
    parser.add_argument("--joint-train-epochs", type=int, default=None)
    parser.add_argument("--out-encoder-ckpt", type=str, default=None)
    parser.add_argument("--pretrained-encoder-ckpt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None, help="Global random seed.")
    parser.add_argument(
        "--clean-ssl-mode",
        action="store_true",
        help="Force clean SSL mode even for custom recipes.",
    )
    parser.add_argument(
        "--disable-generative-save",
        action="store_true",
        help="Disable CSV generation/export after training.",
    )
    parser.add_argument(
        "--set",
        dest="config_overrides",
        action="append",
        default=None,
        help="Config override in key=value form. Repeat to set multiple fields.",
    )
    return parser.parse_args(argv)


def build_config_from_args(args):
    config = apply_training_recipe(Config(), getattr(args, "recipe", "custom"))

    if getattr(args, "data_path", None):
        config.data_path = args.data_path
    if getattr(args, "data_contract", None):
        config.data_contract_path = args.data_contract
    if getattr(args, "create_data_contract", False):
        config.create_data_contract_if_missing = True
    if getattr(args, "accelerator", None):
        config.trainer_accelerator = args.accelerator
    if getattr(args, "devices", None) is not None:
        config.trainer_devices = args.devices
    if getattr(args, "representation_pretrain_epochs", None) is not None:
        config.representation_pretrain_epochs = args.representation_pretrain_epochs
    if getattr(args, "generator_train_epochs", None) is not None:
        config.generator_train_epochs = args.generator_train_epochs
    if getattr(args, "joint_train_epochs", None) is not None:
        config.joint_train_epochs = args.joint_train_epochs
    if getattr(args, "out_encoder_ckpt", None):
        config.out_encoder_ckpt = args.out_encoder_ckpt
    if getattr(args, "pretrained_encoder_ckpt", None):
        config.pretrained_encoder_ckpt = args.pretrained_encoder_ckpt
    if getattr(args, "seed", None) is not None:
        config.seed = args.seed
    if getattr(args, "clean_ssl_mode", False):
        config.clean_ssl_mode = True
    if getattr(args, "disable_generative_save", False):
        config.use_generative_save = False

    config = apply_config_overrides(config, getattr(args, "config_overrides", None))
    return apply_runtime_config_overrides(config)


def build_stage1_config(config):
    stage_cfg = copy.deepcopy(config)
    if not getattr(stage_cfg, "allow_stage1_token_prediction_head", False):
        stage_cfg.use_token_prediction_head = False
    stage_cfg.use_diffusion = False
    stage_cfg.epochs = config.representation_pretrain_epochs
    stage_cfg.current_stage = "stage1"
    return apply_runtime_config_overrides(stage_cfg)


def resolve_checkpoint_policy(cfg):
    # The in-module CV probe uses training batches, so it is not a validation
    # metric. Frozen-split evaluation performs external checkpoint selection.
    monitor = cfg.checkpoint_monitor or "loss"
    mode = cfg.checkpoint_mode or "min"
    return monitor, mode


def build_stage_callbacks(cfg, stage_name):
    checkpoint_dir = Path(cfg.checkpoint_dirpath)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    callbacks = [
        RichProgressBar(refresh_rate=1),
        RichModelSummary(max_depth=2),
    ]
    checkpoint_monitor, checkpoint_mode = resolve_checkpoint_policy(cfg)

    if cfg.checkpoint_save_top_k > 0:
        callbacks.append(
            ModelCheckpoint(
                monitor=checkpoint_monitor,
                dirpath=str(checkpoint_dir),
                filename=f"{stage_name}-best",
                save_top_k=cfg.checkpoint_save_top_k,
                save_last=cfg.checkpoint_save_last,
                mode=checkpoint_mode,
            )
        )
    elif cfg.checkpoint_save_last:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename=f"{stage_name}-last",
                save_top_k=0,
                save_last=True,
            )
        )

    if cfg.checkpoint_every_n_epochs > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename=f"{stage_name}-epoch{{epoch:02d}}",
                save_top_k=-1,
                every_n_epochs=cfg.checkpoint_every_n_epochs,
                save_on_train_epoch_end=True,
            )
        )

    return callbacks, checkpoint_monitor, checkpoint_mode


def main(argv=None):
    args = parse_args([] if argv is None else argv)

    # Initialize configuration
    config = build_config_from_args(args)
    if getattr(config, "clean_ssl_mode", False):
        config.generator_train_epochs = 0
    pl.seed_everything(config.seed, workers=True)

    print(f'Using training recipe: {config.train_recipe}')
    print('Preparing Data')
    train_dataset, train_dataloader, eval_dataset, eval_dataloader, config, dataset = prepare_data(config)
    config = apply_runtime_config_overrides(config)
    if config.mean_target_baseline_rmse is not None:
        print(
            f"Mean-target baseline RMSE (cost space): "
            f"{config.mean_target_baseline_rmse:.4f}"
        )

    if config.use_diffusion and getattr(config, "pretrain_diffusion", True):
        diffusion_model = DiffusionModel(config)
        diffusion_trainer = pl.Trainer(
            max_epochs=getattr(config, "pretrain_diffusion_epochs", config.epochs),
            accelerator=config.trainer_accelerator,
            devices=config.trainer_devices,
            logger=pl.loggers.TensorBoardLogger("tb_logs", name="diffusion"),
            callbacks=[RichProgressBar(refresh_rate=1)],
            log_every_n_steps=3
        )
        diffusion_trainer.fit(diffusion_model, train_dataloader)

    def train_stage(cfg, stage_name, ckpt_path=None, freeze=False):
        cfg = apply_runtime_config_overrides(cfg)
        cfg.current_stage = stage_name
        print(f'Starting {stage_name} for {cfg.epochs} epochs')
        if ckpt_path:
            model = load_claims_model_checkpoint(
                HierarchicalClaimsModel,
                ckpt_path,
                config=cfg,
                allow_legacy=cfg.allow_legacy_checkpoint_loading,
                stage_transition=stage_name == "stage2",
            )
        else:
            model = HierarchicalClaimsModel(cfg)

        if cfg.current_stage == "stage1":
            try:
                params = list(model.context_encoder_lvl2.parameters())
                assert any(p.requires_grad for p in params), (
                    "Encoder must be trainable in stage1"
                )
            except Exception:
                pass

        if freeze and cfg.current_stage == "stage2" and cfg.freeze_encoder_at_stage2:
            model.freeze_encoder(getattr(cfg, "encoder_unfreeze_layers", 1))

        if getattr(cfg, "debug_low_threshold", False):
            model.threshold.data = torch.tensor(0.05)
            model.lambda_entropy.data = torch.tensor(0.0)

        callbacks, checkpoint_monitor, checkpoint_mode = build_stage_callbacks(
            cfg,
            stage_name,
        )
        print(
            f"{stage_name} checkpoint policy: "
            f"monitor={checkpoint_monitor} ({checkpoint_mode}), "
            f"top_k={cfg.checkpoint_save_top_k}, "
            f"save_last={cfg.checkpoint_save_last}, "
            f"every_n_epochs={cfg.checkpoint_every_n_epochs}, "
            f"dir={cfg.checkpoint_dirpath}, "
            f"final_ckpt={getattr(cfg, 'out_encoder_ckpt', 'n/a')}"
        )
        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator=cfg.trainer_accelerator,
            devices=cfg.trainer_devices,
            logger=pl.loggers.TensorBoardLogger("tb_logs", name=stage_name),
            callbacks=callbacks,
            log_every_n_steps=3
        )

        if cfg.use_lr_find:
            print('Finding learning rate')
            tuner = pl.tuner.Tuner(trainer)
            lr_finder = tuner.lr_find(model, train_dataloader)
            suggested_lr = lr_finder.suggestion()
            print('Using lr: ', suggested_lr)
        else:
            model.lr = cfg.lr

        trainer.fit(model, train_dataloader)
        return model, trainer

    ckpt_path = None
    model = None

    if getattr(config, "representation_pretrain_epochs", 0) > 0:
        stage_cfg = build_stage1_config(config)
        model, trainer = train_stage(stage_cfg, "stage1")
        ckpt_path = stage_cfg.out_encoder_ckpt
        trainer.save_checkpoint(ckpt_path)

    if getattr(config, "generator_train_epochs", 0) > 0:
        stage_cfg = copy.deepcopy(config)
        stage_cfg.clean_ssl_mode = False
        stage_cfg.use_token_prediction_head = True
        stage_cfg.use_diffusion = True
        stage_cfg.epochs = config.generator_train_epochs
        stage_cfg.current_stage = "stage2"
        stage_cfg.sae_weight = 0
        stage_cfg.level_2_weight = 0
        stage_cfg = apply_runtime_config_overrides(stage_cfg)
        ckpt_to_load = stage_cfg.pretrained_encoder_ckpt if stage_cfg.pretrained_encoder_ckpt else ckpt_path
        if not ckpt_to_load or not os.path.exists(ckpt_to_load):
            raise FileNotFoundError(
                f"Stage2 requires encoder checkpoint at {ckpt_to_load}"
            )
        model, trainer = train_stage(stage_cfg, "stage2", ckpt_path=ckpt_to_load, freeze=True)
        ckpt_path = "generator_stage.ckpt"
        trainer.save_checkpoint(ckpt_path)

    if model is None:
        # Fallback to single stage training with current config
        config.current_stage = "joint"
        config = apply_runtime_config_overrides(config)
        model, trainer = train_stage(config, "jepa")
        ckpt_path = "final.ckpt"
        trainer.save_checkpoint(ckpt_path)

    if getattr(config, "joint_train_epochs", 0) > 0:
        stage_cfg = copy.deepcopy(config)
        stage_cfg.use_token_prediction_head = True
        stage_cfg.use_diffusion = True
        stage_cfg.epochs = config.joint_train_epochs
        stage_cfg.current_stage = "joint"
        stage_cfg = apply_runtime_config_overrides(stage_cfg)
        model, trainer = train_stage(stage_cfg, "joint", ckpt_path=ckpt_path)
        ckpt_path = "joint.ckpt"
        trainer.save_checkpoint(ckpt_path)

    # === Generation Block Start ===
    # Generate predictions if either the token prediction head or diffusion
    # generator is enabled. This allows saving predictions when the model
    # relies solely on diffusion-based generation.
    if config.use_generative_save and (
        config.use_token_prediction_head or config.use_diffusion
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        evaluation_dataloader = DataLoader(
            eval_dataset,
            batch_size=128,
            collate_fn=dataset.collate_eval_fn,
            shuffle=False
        )
        results = []

        with torch.no_grad():
            for batch in tqdm(evaluation_dataloader):
                cpt_tensor, icd_tensor, ttnc_tensor, target = batch
                cpt_tensor = cpt_tensor.to(device)
                icd_tensor = icd_tensor.to(device)
                ttnc_tensor = ttnc_tensor.to(device)
                target = target.to(device)

                # Invoke the model with generation mode enabled
                outputs = model(
                    cpt_tensor=cpt_tensor,
                    icd_tensor=icd_tensor,
                    ttnc_tensor=ttnc_tensor,
                    generation=True,  # Enable generation mode
                    teacher_forcing=False
                )

                # Extract predictions
                predicted_cpt_codes = outputs['predicted_cpt_codes']
                predicted_icd_codes = outputs['predicted_icd_codes']
                predicted_ttnc_code = outputs['predicted_ttnc_code']

                decoded_cpt = decode_predicted_codes(
                    predicted_cpt_codes, config.cpt_id_to_token
                )
                decoded_icd = decode_predicted_codes(
                    predicted_icd_codes, config.icd_id_to_token
                )

                for i in range(cpt_tensor.size(0)):
                    predicted_cpt_codes_list = decoded_cpt[i]

                    # Process actual CPT codes (from the last claim)
                    actual_cpt_indices = cpt_tensor[i, -1, :].cpu().numpy()
                    actual_cpt_codes = [config.cpt_id_to_token.get(idx, '<UNK>') for idx in actual_cpt_indices if idx != 0]

                    predicted_icd_codes_list = decoded_icd[i]

                    # Process actual ICD codes (from the last claim)
                    actual_icd_indices = icd_tensor[i, -1, :].cpu().numpy()
                    actual_icd_codes = [config.icd_id_to_token.get(idx, '<UNK>') for idx in actual_icd_indices if idx != 0]

                    # Process predicted TTNC code
                    predicted_ttnc_idx = predicted_ttnc_code[i].item()
                    predicted_ttnc_code_str = config.ttnc_id_to_token.get(predicted_ttnc_idx, '<UNK>')

                    # Process actual TTNC code
                    actual_ttnc_idx = ttnc_tensor[i, -1].item()
                    actual_ttnc_code = config.ttnc_id_to_token.get(actual_ttnc_idx, '<UNK>') if actual_ttnc_idx != 0 else ''

                    # Compile results
                    result = {
                        'predicted_cpt': ' '.join(predicted_cpt_codes_list),
                        'actual_cpt': ' '.join(actual_cpt_codes),
                        'predicted_icd': ' '.join(predicted_icd_codes_list),
                        'actual_icd': ' '.join(actual_icd_codes),
                        'predicted_ttnc': predicted_ttnc_code_str,
                        'actual_ttnc': actual_ttnc_code,
                        'target': target[i].item()
                    }
                    results.append(result)

        df = pd.DataFrame(results)
        df.to_csv('predictions.csv', index=False)

    # === Plotting Block (Unchanged) ===
    if config.use_plotting:
        model.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for batch in train_dataloader:
                cpt_tensor, icd_tensor, ttnc_tensor, labels = batch  # Adjust this line based on your data structure
                outputs = model(
                    cpt_tensor=cpt_tensor,
                    icd_tensor=icd_tensor,
                    ttnc_tensor=ttnc_tensor,
                    target=labels,  # Pass target if needed for embeddings
                    generation=False  # Ensure training_forward is used
                )
                embeddings = outputs['patient_representation']
                all_embeddings.append(embeddings.cpu().numpy())
                labels_exp = torch.exp(labels)
                all_labels.append(labels_exp.cpu().numpy())

        all_embeddings = np.concatenate(all_embeddings, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        def bucket_labels(label):
                if label < 1500:
                    return '< 1500'
                elif 1500 <= label <= 4500:
                    return '1500-4500'
                elif 4500 <= label <= 7500:
                    return '4500-7500'
                else:
                    return '> 7500'

        # Apply bucketing to all_labels
        all_labels_buckets = np.array([bucket_labels(label) for label in all_labels])

        # Step 2: Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, init='pca', perplexity=50)
        embeddings_2d = tsne.fit_transform(all_embeddings)

        # Step 3: Visualize
        df = pd.DataFrame(embeddings_2d, columns=['Dim1', 'Dim2'])
        df['label'] = all_labels_buckets  # Use bucketed labels

        plt.figure(figsize=(8, 6))
        sns.scatterplot(x='Dim1', y='Dim2', hue='label', palette='tab10', data=df, s=60, alpha=0.7)
        plt.title("t-SNE Visualization of Embeddings with Bucketed Labels")
        plt.xlabel("t-SNE Dim 1")
        plt.ylabel("t-SNE Dim 2")
        plt.legend(title='Class')
        plt.show()
    # === Plotting Block End ===


if __name__ == '__main__':
    main(sys.argv[1:])
