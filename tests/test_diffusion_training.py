import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from jepa_utils.config import Config
from models.diffusion import ClaimD3PM

class ConstantDataset(Dataset):
    def __init__(self, num_samples, seq_len, token=1):
        self.data = torch.full((num_samples, seq_len), token, dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

class TestDiffusionTraining(unittest.TestCase):
    def test_single_epoch_reduces_ce(self):
        cfg = Config()
        cfg.cpt_vocab_size = 3
        cfg.icd_vocab_size = 3
        cfg.ttnc_vocab_size = 2
        cfg.embedding_dim = 4
        cfg.diffusion_steps = 5
        vocab_size = cfg.cpt_vocab_size + cfg.icd_vocab_size + cfg.ttnc_vocab_size + 3
        dataset = ConstantDataset(20, 6)
        loader = DataLoader(dataset, batch_size=4)
        model = ClaimD3PM(cfg, vocab_size, condition_dim=cfg.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model.train()
        for tokens in loader:
            optimizer.zero_grad()
            loss = model(tokens)
            loss.backward()
            optimizer.step()
        # Evaluate cross-entropy after one epoch
        model.eval()
        with torch.no_grad():
            losses = [model(batch) for batch in loader]
        avg_loss = torch.stack(losses).mean()
        self.assertLess(avg_loss.item(), 2.0)

if __name__ == '__main__':
    unittest.main()
