import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from utils.config import Config
from models.data_prep import prepare_data
from models.hierarchical_model import HierarchicalClaimsModel
from models.sparse_autoencoder import SparseAutoencoder


def extract_embeddings(model, dataloader, device):
    """Run the hierarchical model to obtain patient representations."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            cpt_tensor, icd_tensor, ttnc_tensor, _ = batch
            cpt_tensor = cpt_tensor.to(device)
            icd_tensor = icd_tensor.to(device)
            ttnc_tensor = ttnc_tensor.to(device)
            outputs = model(cpt_tensor=cpt_tensor, icd_tensor=icd_tensor, ttnc_tensor=ttnc_tensor)
            embeddings.append(outputs["patient_representation"].cpu())
    return torch.cat(embeddings, dim=0)


def train_autoencoder(embeddings, hidden_dim=64, k=32, epochs=10, lr=1e-3, device="cpu"):
    dataset = TensorDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
    sae = SparseAutoencoder(embeddings.size(1), hidden_dim, k).to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    sae.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for (batch,) in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = sae(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch.size(0)
        avg_loss = running_loss / len(dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
    return sae


def main():
    config = Config()
    # Load data
    _, train_dataloader, _, _, config, _ = prepare_data(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained hierarchical model
    model = HierarchicalClaimsModel(config)
    model.load_state_dict(torch.load("hierarchical_model.pt", map_location=device))
    model = model.to(device)

    # Extract embeddings
    embeddings = extract_embeddings(model, train_dataloader, device)

    # Train SAE
    sae = train_autoencoder(embeddings, hidden_dim=config.embedding_dim, k=32, epochs=20, device=device)

    # Save weights
    torch.save(sae.state_dict(), "sparse_autoencoder.pt")
    print("Saved sparse autoencoder weights to sparse_autoencoder.pt")


if __name__ == "__main__":
    main()
