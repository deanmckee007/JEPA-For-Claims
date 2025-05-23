import unittest
import torch
from jepa_utils.config import Config
from jepa_models.diffusion import DiffusionModel
from jepa_models.discrete_diffusion import DiscreteDiffusionModel

class TestDiffusionModel(unittest.TestCase):
    def test_forward_and_sample(self):
        config = Config()
        config.cpt_vocab_size = 10
        config.icd_vocab_size = 10
        config.ttnc_vocab_size = 5
        config.embedding_dim = 4
        config.diffusion_steps = 5
        model = DiffusionModel(config)
        cpt_tokens = torch.randint(0, 10, (2, config.max_cpt_tokens))
        icd_tokens = torch.randint(0, 10, (2, config.max_icd_tokens))
        ttnc_tokens = torch.randint(0, 5, (2,))
        loss = model(cpt_tokens, icd_tokens, ttnc_tokens)
        self.assertTrue(loss.dim() == 0)
        cpt_pred, icd_pred, ttnc_pred = model.sample(2)
        self.assertEqual(cpt_pred.shape, (2,))
        self.assertEqual(icd_pred.shape, (2,))
        self.assertEqual(ttnc_pred.shape, (2,))

    def test_discrete_diffusion_forward(self):
        config = Config()
        config.cpt_vocab_size = 10
        config.icd_vocab_size = 10
        config.ttnc_vocab_size = 5
        config.embedding_dim = 4
        config.diffusion_steps = 5
        model = DiscreteDiffusionModel(config)
        cpt_tokens = torch.randint(0, 10, (2, config.max_cpt_tokens))
        icd_tokens = torch.randint(0, 10, (2, config.max_icd_tokens))
        ttnc_tokens = torch.randint(0, 5, (2,))
        loss = model(cpt_tokens, icd_tokens, ttnc_tokens)
        self.assertTrue(loss.dim() == 0)
        cpt_pred, icd_pred, ttnc_pred = model.sample(2)
        self.assertEqual(cpt_pred.shape, (2, config.max_cpt_tokens))
        self.assertEqual(icd_pred.shape, (2, config.max_icd_tokens))
        self.assertEqual(ttnc_pred.shape, (2,))

    def test_conditioned_sample(self):
        config = Config()
        config.cpt_vocab_size = 10
        config.icd_vocab_size = 10
        config.ttnc_vocab_size = 5
        config.embedding_dim = 4
        config.diffusion_steps = 5
        model = DiffusionModel(config, condition_dim=4)
        condition = torch.randn(2, 4)
        cpt_pred, icd_pred, ttnc_pred = model.sample(2, condition=condition)
        self.assertEqual(cpt_pred.shape, (2,))
        self.assertEqual(icd_pred.shape, (2,))
        self.assertEqual(ttnc_pred.shape, (2,))

if __name__ == '__main__':
    unittest.main()
