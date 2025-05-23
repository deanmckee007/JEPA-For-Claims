# scripts/train.py
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, RichModelSummary
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from jepa_models.data_prep import prepare_data
from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_models.diffusion import DiffusionModel
from jepa_utils.tensor_utils import calculate_entropy, adaptive_sampling
from jepa_utils.metrics import calculate_rmse
from jepa_utils.config import Config


def main():
    # Initialize configuration
    config = Config()

    print('Preparing Data')
    train_dataset, train_dataloader, eval_dataset, eval_dataloader, config, dataset = prepare_data(config)

    if config.use_diffusion and getattr(config, "pretrain_diffusion", True):
        diffusion_model = DiffusionModel(config)
        diffusion_trainer = pl.Trainer(
            max_epochs=config.epochs,
            accelerator='gpu',
            logger=pl.loggers.TensorBoardLogger("tb_logs", name="diffusion"),
            callbacks=[RichProgressBar(refresh_rate=1)]
        )
        diffusion_trainer.fit(diffusion_model, train_dataloader)

    # Initialize model
    print('max claims len', config.max_claims_len)
    print(f'Using {config.epochs} epochs')
    try:
        model = HierarchicalClaimsModel(config)
        print("Model instantiated successfully")
    except Exception as e:
        print(f"Error during model instantiation: {e}")

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=config.epochs,
        accelerator='gpu',
        # gradient_clip_val=10.0,
        logger=pl.loggers.TensorBoardLogger("tb_logs", name="jepa"),
        callbacks=[
            #pl.callbacks.EarlyStopping(monitor='val_rmse', patience=10, mode='min'),
            RichProgressBar(refresh_rate=1),# 1 = update every batch
            RichModelSummary(max_depth=2),
            pl.callbacks.ModelCheckpoint(
                monitor='val_rmse',
                dirpath='checkpoints/',
                filename='best-checkpoint',
                save_top_k=1,
                mode='min'
            )
        ]
    )

    if config.use_lr_find:
        print('Finding learning rate')
        tuner = pl.tuner.Tuner(trainer)
        lr_finder = tuner.lr_find(model, train_dataloader)
        suggested_lr = lr_finder.suggestion()
        print('Using lr: ', suggested_lr)
    else:
        model.lr = config.lr

    print('Training')
    # Train the model
    trainer.fit(model, train_dataloader)

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

                for i in range(cpt_tensor.size(0)):
                    # Process predicted CPT codes
                    if predicted_cpt_codes.dim() == 2:
                        pred_cpt_idx = (predicted_cpt_codes[i] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                        predicted_cpt_codes_list = [config.cpt_id_to_token.get(idx, '<UNK>') for idx in pred_cpt_idx if idx != 0]
                    else:
                        pred_cpt_idx = predicted_cpt_codes[i].item()
                        predicted_cpt_codes_list = [config.cpt_id_to_token.get(pred_cpt_idx, '<UNK>')]

                    # Process actual CPT codes (from the last claim)
                    actual_cpt_indices = cpt_tensor[i, -1, :].cpu().numpy()
                    actual_cpt_codes = [config.cpt_id_to_token.get(idx, '<UNK>') for idx in actual_cpt_indices if idx != 0]

                    # Process predicted ICD codes
                    if predicted_icd_codes.dim() == 2:
                        pred_icd_idx = (predicted_icd_codes[i] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                        predicted_icd_codes_list = [config.icd_id_to_token.get(idx, '<UNK>') for idx in pred_icd_idx if idx != 0]
                    else:
                        pred_icd_idx = predicted_icd_codes[i].item()
                        predicted_icd_codes_list = [config.icd_id_to_token.get(pred_icd_idx, '<UNK>')]

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
    main()
