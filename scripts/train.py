# scripts/train.py
import torch
import pytorch_lightning as pl
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from models.data_prep import prepare_data
from models.hierarchical_model import HierarchicalClaimsModel
from utils.metrics import calculate_rmse
from utils.config import Config


def main():
    # Initialize configuration
    config = Config()

    print('Preparing Data')
    dataset, dataloader, config = prepare_data(config)

    # Initialize model
    print('max claims len', config.max_claims_len)
    try:
        model = HierarchicalClaimsModel(config)
        print("Model instantiated successfully")
    except Exception as e:
        print(f"Error during model instantiation: {e}")

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator='gpu',
        gradient_clip_val=3.0,
        logger=pl.loggers.TensorBoardLogger("tb_logs", name="jepa"),
        callbacks=[
            pl.callbacks.EarlyStopping(monitor='Iloss2', patience=10, mode='min'),
            pl.callbacks.ModelCheckpoint(
                monitor='val_rmse',
                dirpath='checkpoints/',
                filename='best-checkpoint',
                save_top_k=1,
                mode='min'
            )
        ]
    )

    print('Finding learning rate')
    tuner = pl.tuner.Tuner(trainer)
    # lr_finder = tuner.lr_find(model, dataloader)
    # suggested_lr = lr_finder.suggestion()
    # print('Using lr: ', suggested_lr)

    model.lr = 1e-3

    print('Training')
    # Train the model
    trainer.fit(model, dataloader)

    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            cpt_tensor, icd_tensor, ttnc_tensor, labels = batch  # Adjust this line based on your data structure
            outputs = model(cpt_tensor, icd_tensor, ttnc_tensor)
            embeddings = outputs['patient_representation']
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Step 2: Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(all_embeddings)

    # Step 3: Visualize
    df = pd.DataFrame(embeddings_2d, columns=['Dim1', 'Dim2'])
    df['label'] = all_labels

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='Dim1', y='Dim2', hue='label', palette='tab10', data=df, s=60, alpha=0.7)
    plt.title("t-SNE Visualization of Embeddings")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    plt.legend(title='Class')
    plt.show()

if __name__ == '__main__':
    main()
