import unittest
import torch
from jepa_utils.config import Config
from jepa_models.diffusion import DiffusionModel


class TestDiffusionConditionGradient(unittest.TestCase):
    def test_condition_gradient_flow(self):
        config = Config()
        config.cpt_vocab_size = 10
        config.icd_vocab_size = 10
        config.ttnc_vocab_size = 5
        config.embedding_dim = 4
        config.diffusion_steps = 5

        model = DiffusionModel(config, condition_dim=4)
        batch_size = 2
        cpt_tokens = torch.randint(0, config.cpt_vocab_size, (batch_size, config.max_cpt_tokens))
        icd_tokens = torch.randint(0, config.icd_vocab_size, (batch_size, config.max_icd_tokens))
        ttnc_tokens = torch.randint(0, config.ttnc_vocab_size, (batch_size,))
        condition = torch.randn(batch_size, 4, requires_grad=True)

        loss = model(cpt_tokens, icd_tokens, ttnc_tokens, condition=condition)
        loss.backward()

        self.assertIsNotNone(condition.grad)
        self.assertGreater(condition.grad.norm().item(), 0)


if __name__ == '__main__':
    unittest.main()
