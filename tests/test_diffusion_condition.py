import unittest
import torch
from jepa_utils.config import Config
from jepa_models.diffusion import DiffusionModel

class TestDiffusionCondition(unittest.TestCase):
    def test_condition_gradient_flow(self):
        torch.manual_seed(0)
        config = Config()
        config.cpt_vocab_size = 10
        config.icd_vocab_size = 10
        config.ttnc_vocab_size = 5
        config.embedding_dim = 4
        config.diffusion_steps = 5
        model = DiffusionModel(config, condition_dim=4)
        cpt_tokens = torch.randint(0, config.cpt_vocab_size, (2, config.max_cpt_tokens))
        icd_tokens = torch.randint(0, config.icd_vocab_size, (2, config.max_icd_tokens))
        ttnc_tokens = torch.randint(0, config.ttnc_vocab_size, (2,))
        condition = torch.randn(2, 4, requires_grad=True)
        loss = model(cpt_tokens, icd_tokens, ttnc_tokens, condition=condition)
        loss.backward()
        self.assertIsNotNone(condition.grad)
        self.assertFalse(torch.allclose(condition.grad, torch.zeros_like(condition.grad)))

if __name__ == '__main__':
    unittest.main()
