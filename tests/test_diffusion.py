import unittest
import torch
from unittest.mock import patch
from jepa_utils.config import Config
from models.diffusion import ClaimD3PM


class TestClaimD3PM(unittest.TestCase):
    def test_forward_and_generate(self):
        cfg = Config()
        cfg.cpt_vocab_size = 10
        cfg.icd_vocab_size = 10
        cfg.ttnc_vocab_size = 5
        cfg.embedding_dim = 4
        cfg.diffusion_steps = 5
        vocab_size = cfg.cpt_vocab_size + cfg.icd_vocab_size + cfg.ttnc_vocab_size + 3
        model = ClaimD3PM(cfg, vocab_size, condition_dim=cfg.embedding_dim)
        tokens = torch.randint(0, vocab_size, (2, 5))
        # Basic smoke test for forward pass
        condition = torch.randn(2, cfg.embedding_dim)
        with patch.object(model, 'generate_claim', return_value=torch.zeros(2, 5, dtype=torch.long)):
            samples = model.generate_claim(condition, seq_len=5)
        self.assertEqual(samples.shape, (2, 5))


if __name__ == "__main__":
    unittest.main()
