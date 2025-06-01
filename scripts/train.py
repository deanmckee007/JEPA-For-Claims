# scripts/train.py
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from jepa_utils.data_prep import prepare_data
from jepa_models.hierarchical_model import HierarchicalClaimsModel
from diffusion_models import ClaimD3PM
from jepa_utils.tensor_utils import calculate_entropy, adaptive_sampling
from jepa_utils.metrics import calculate_rmse
from jepa_utils.config import Config
from jepa_utils.prediction_utils import decode_predicted_codes
import copy


def main(argv=None):
    if argv is None:
        argv = []
    parser = argparse.ArgumentParser(description="JEPA training")
    parser.add_argument("--phase", choices=["pretrain", "joint"], default=None)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume for joint phase")
    args = parser.parse_args(argv)

    # Initialize configuration
    config = Config()

    print('Preparing Data')
    train_dataset, train_dataloader, eval_dataset, eval_dataloader, config, dataset = prepare_data(config)

    if args.phase == "pretrain":
        if config.use_diffusion:
            vocab_size = (
                config.cpt_vocab_size + config.icd_vocab_size + config.ttnc_vocab_size + 3
            )
            diffusion_model = ClaimD3PM(config, vocab_size, config.output_dim)
            diffusion_trainer = pl.Trainer(
                max_epochs=getattr(config, "pretrain_diffusion_epochs", config.epochs),
                accelerator='gpu',
                logger=pl.loggers.TensorBoardLogger("tb_logs", name="diffusion"),
                callbacks=[RichProgressBar(refresh_rate=1)],
                log_every_n_steps=3
            )
            diffusion_trainer.fit(diffusion_model, train_dataloader)
            diffusion_trainer.save_checkpoint("diffusion_only.ckpt")
        return

    if args.phase == "joint":
        ckpt = args.resume or "diffusion_only.ckpt"
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Diffusion checkpoint {ckpt} not found")
        vocab_size = (
            config.cpt_vocab_size
            + config.icd_vocab_size
            + config.ttnc_vocab_size
            + 3
        )
        diffusion_model = ClaimD3PM.load_from_checkpoint(
            ckpt,
            config=config,
            vocab_size=vocab_size,
            condition_dim=config.output_dim,
        )
        config.use_diffusion = True
        config.current_stage = "joint"
        model = HierarchicalClaimsModel(config, diffusion=diffusion_model)
        trainer = pl.Trainer(
            max_epochs=config.epochs,
            accelerator='gpu',
            logger=pl.loggers.TensorBoardLogger("tb_logs", name="joint"),
            callbacks=[RichProgressBar(refresh_rate=1), RichModelSummary(max_depth=2)],
            log_every_n_steps=3,
        )
        trainer.fit(model, train_dataloader)
        trainer.save_checkpoint("joint.ckpt")
        return

    if args.phase is None and config.use_diffusion and getattr(config, "pretrain_diffusion", True):
        vocab_size = (
            config.cpt_vocab_size + config.icd_vocab_size + config.ttnc_vocab_size + 3
        )
        diffusion_model = ClaimD3PM(config, vocab_size, config.output_dim)
        diffusion_trainer = pl.Trainer(
            max_epochs=getattr(config, "pretrain_diffusion_epochs", config.epochs),
            accelerator='gpu',
            logger=pl.loggers.TensorBoardLogger("tb_logs", name="diffusion"),
            callbacks=[RichProgressBar(refresh_rate=1)],
            log_every_n_steps=3
        )
        diffusion_trainer.fit(diffusion_model, train_dataloader)
        diffusion_trainer.save_checkpoint("diffusion_only.ckpt")

    def train_stage(cfg, stage_name, ckpt_path=None, freeze=False):
        cfg.current_stage = stage_name
        print(f'Starting {stage_name} for {cfg.epochs} epochs')
        if ckpt_path:
            model = HierarchicalClaimsModel.load_from_checkpoint(
                ckpt_path, config=cfg, strict=False
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

        trainer = pl.Trainer(
            max_epochs=cfg.epochs,
            accelerator='gpu',
            logger=pl.loggers.TensorBoardLogger("tb_logs", name=stage_name),
            callbacks=[
                RichProgressBar(refresh_rate=1),
                RichModelSummary(max_depth=2),
                pl.callbacks.ModelCheckpoint(
                    monitor='val_rmse',
                    dirpath='checkpoints/',
                    filename=f'{stage_name}-best',
                    save_top_k=1,
                    mode='min'
                )
            ],
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
        stage_cfg = copy.deepcopy(config)
        stage_cfg.use_token_prediction_head = False
        stage_cfg.use_diffusion = False
        stage_cfg.epochs = config.representation_pretrain_epochs
        stage_cfg.current_stage = "stage1"
        model, trainer = train_stage(stage_cfg, "stage1")
        ckpt_path = stage_cfg.out_encoder_ckpt
        trainer.save_checkpoint(ckpt_path)

    if getattr(config, "generator_train_epochs", 0) > 0:
        stage_cfg = copy.deepcopy(config)
        stage_cfg.use_diffusion = True
        stage_cfg.epochs = config.generator_train_epochs
        stage_cfg.current_stage = "stage2"
        stage_cfg.sae_weight = 0
        stage_cfg.level_2_weight = 0
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
        model, trainer = train_stage(config, "jepa")
        ckpt_path = "final.ckpt"
        trainer.save_checkpoint(ckpt_path)

    if getattr(config, "joint_train_epochs", 0) > 0:
        stage_cfg = copy.deepcopy(config)
        stage_cfg.use_token_prediction_head = True
        stage_cfg.use_diffusion = True
        stage_cfg.epochs = config.joint_train_epochs
        stage_cfg.current_stage = "joint"
        model, trainer = train_stage(stage_cfg, "joint", ckpt_path=ckpt_path)
        ckpt_path = "joint.ckpt"
        trainer.save_checkpoint(ckpt_path)

    # === Generation Block Start ===
    # Generate predictions if either the token prediction head or diffusion
    # generator is enabled. This allows saving predictions when the model
    # relies solely on diffusion-based generation.
    if config.use_generative_save and (
        getattr(model, "use_token_prediction_head", False)
        or getattr(model, "use_diffusion", False)
    ):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        evaluation_dataloader = DataLoader(
            eval_dataset,
            batch_size=128,
            collate_fn=dataset.collate_fn,
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
    import sys
    main(sys.argv[1:])
