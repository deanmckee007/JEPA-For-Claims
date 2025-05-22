import unittest
import torch
from jepa_utils.config import Config
from jepa_models.hierarchical_model import HierarchicalClaimsModel

class TestDiffusionIntegration(unittest.TestCase):
    def test_autoregressive_generation_with_diffusion(self):
        config = Config()
        config.use_diffusion = True
        config.use_sparse_autoencoder = False
        config.use_token_prediction_head = False
        config.use_predictor_head = False
        # Optional rarity score tensors expected by the model
        config.cpt_rarity_scores = None
        config.icd_rarity_scores = None
        config.ttnc_rarity_scores = None
        config.cpt_vocab_size = 10
        config.icd_vocab_size = 10
        config.ttnc_vocab_size = 5
        config.embedding_dim = 4
        config.max_cpt_tokens = 3
        config.max_icd_tokens = 3
        config.max_claims_len = 2
        config.diffusion_steps = 5

        model = HierarchicalClaimsModel(config)
        batch_size = 2
        cpt_tensor = torch.randint(0, config.cpt_vocab_size, (batch_size, config.max_claims_len, config.max_cpt_tokens))
        icd_tensor = torch.randint(0, config.icd_vocab_size, (batch_size, config.max_claims_len, config.max_icd_tokens))
        ttnc_tensor = torch.randint(0, config.ttnc_vocab_size, (batch_size, config.max_claims_len))

        outputs = model.autoregressive_generation(cpt_tensor, icd_tensor, ttnc_tensor)
        self.assertEqual(outputs['predicted_cpt_codes'].shape, (batch_size,))
        self.assertEqual(outputs['predicted_icd_codes'].shape, (batch_size,))
        self.assertEqual(outputs['predicted_ttnc_code'].shape, (batch_size,))

if __name__ == '__main__':
    unittest.main()
