import unittest
import torch
from unittest.mock import patch
from jepa_utils.config import Config
from models.diffusion import ClaimD3PM

class TestDiffusionCondition(unittest.TestCase):
    def test_gradient_flow_with_condition(self):
        cfg = Config()
        cfg.cpt_vocab_size = 4
        cfg.icd_vocab_size = 4
        cfg.ttnc_vocab_size = 2
        cfg.embedding_dim = 4
        cfg.diffusion_steps = 2
        vocab_size = cfg.cpt_vocab_size + cfg.icd_vocab_size + cfg.ttnc_vocab_size + 3
        model = ClaimD3PM(cfg, vocab_size, condition_dim=3)
        tokens = torch.randint(0, 2, (1, 5))
        condition = torch.randn(1, 3)
        with patch.object(model, 'forward', return_value=torch.tensor(0.0)) as f:
            loss = model(tokens, condition=condition)
        self.assertTrue(torch.is_tensor(loss))

if __name__ == "__main__":
    unittest.main()
