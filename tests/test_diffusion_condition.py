import unittest
import torch
from jepa_utils.config import Config
from jepa_models.diffusion import DiffusionModel

class TestDiffusionCondition(unittest.TestCase):
    def test_gradient_flow_with_condition(self):
        cfg = Config()
        cfg.cpt_vocab_size = 4
        cfg.icd_vocab_size = 4
        cfg.ttnc_vocab_size = 2
        cfg.embedding_dim = 3
        cfg.diffusion_steps = 2
        model = DiffusionModel(cfg, condition_dim=3)
        cpt = torch.randint(0, cfg.cpt_vocab_size, (1, cfg.max_cpt_tokens))
        icd = torch.randint(0, cfg.icd_vocab_size, (1, cfg.max_icd_tokens))
        ttnc = torch.randint(0, cfg.ttnc_vocab_size, (1,))
        condition = torch.randn(1, 3)
        loss = model(cpt, icd, ttnc, condition=condition)
        loss.backward()
        total_grad = sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
        self.assertGreater(total_grad, 0)

if __name__ == "__main__":
    unittest.main()
