import unittest
import torch
from torch.utils.data import DataLoader, Dataset
from jepa_utils.config import Config
from models.diffusion import ClaimD3PM

class ToyDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab = vocab
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        torch.manual_seed(idx)
        return torch.randint(1, self.vocab, (self.seq_len,), dtype=torch.long)

class TestDiffusionProgress(unittest.TestCase):
    def test_loss_drops_after_three_epochs(self):
        cfg = Config()
        cfg.cpt_vocab_size = 3
        cfg.icd_vocab_size = 3
        cfg.ttnc_vocab_size = 2
        cfg.embedding_dim = 4
        cfg.diffusion_steps = 5
        vocab_size = cfg.cpt_vocab_size + cfg.icd_vocab_size + cfg.ttnc_vocab_size + 3
        torch.manual_seed(0)
        dataset = ToyDataset(24, 6, vocab_size)
        loader = DataLoader(dataset, batch_size=4)
        model = ClaimD3PM(cfg, vocab_size, condition_dim=cfg.embedding_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.8)
        model.train()
        epoch_losses = []
        for _ in range(3):
            for tokens in loader:
                optimizer.zero_grad()
                loss = model(tokens)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                losses = [model(batch) for batch in loader]
            epoch_losses.append(torch.stack(losses).mean().item())
        self.assertGreaterEqual(epoch_losses[0] - epoch_losses[-1], 0.5)

if __name__ == '__main__':
    unittest.main()
