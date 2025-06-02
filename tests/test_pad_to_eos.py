import unittest
import torch
from jepa_utils.config import Config
from diffusion_models.claim_d3pm import ClaimD3PM

class TestPadToEos(unittest.TestCase):
    def test_generated_length_histogram(self):
        cfg = Config()
        cfg.cpt_vocab_size = 2
        cfg.icd_vocab_size = 1
        cfg.ttnc_vocab_size = 1
        cfg.embedding_dim = 8
        cfg.diffusion_steps = 2
        vocab_size = cfg.cpt_vocab_size + cfg.icd_vocab_size + cfg.ttnc_vocab_size + 3
        model = ClaimD3PM(cfg, vocab_size, cfg.embedding_dim)

        real_lengths = torch.tensor([2,3,4,3,2])
        seq_len = 5
        tokens = torch.zeros(len(real_lengths), seq_len, dtype=torch.long)
        for i,l in enumerate(real_lengths):
            tokens[i,:l] = torch.randint(1, vocab_size, (l,))

        optim = torch.optim.Adam(model.parameters(), lr=0.01)
        for _ in range(5):
            optim.zero_grad()
            t = torch.randint(0, cfg.diffusion_steps, (tokens.size(0),))
            loss = model.p_losses(tokens, t, model.default_condition.expand(tokens.size(0), -1))
            loss.backward()
            optim.step()

        model.eval()
        with torch.no_grad():
            samples = model.generate_claim(model.default_condition.expand(100, -1), seq_len)
        lengths = []
        for row in samples:
            row = row.tolist()
            length = row.index(0) if 0 in row else len(row)
            lengths.append(length)
        lengths = torch.tensor(lengths)
        hist_real = torch.bincount(real_lengths, minlength=seq_len+1).float()
        hist_real /= hist_real.sum()
        hist_gen = torch.bincount(lengths, minlength=seq_len+1).float()
        hist_gen /= hist_gen.sum()
        diff = torch.abs(hist_real - hist_gen)
        self.assertTrue(torch.all(diff <= 0.3))

if __name__ == '__main__':
    unittest.main()
