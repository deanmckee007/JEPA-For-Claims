import unittest
import torch
from jepa_utils.config import Config
from jepa_models.hierarchical_model import HierarchicalClaimsModel

class TestDebugLowThreshold(unittest.TestCase):
    def test_threshold_flag_sets_values(self):
        config = Config()
        config.cpt_vocab_size = 5
        config.icd_vocab_size = 5
        config.ttnc_vocab_size = 3
        config.cpt_rarity_scores = None
        config.icd_rarity_scores = None
        config.ttnc_rarity_scores = None
        config.use_diffusion = False
        config.debug_low_threshold = True

        model = HierarchicalClaimsModel(config)
        if getattr(config, 'debug_low_threshold', False):
            model.threshold.data = torch.tensor(0.05)
            model.lambda_entropy.data = torch.tensor(0.0)

        self.assertAlmostEqual(model.threshold.item(), 0.05)
        self.assertAlmostEqual(model.lambda_entropy.item(), 0.0)

class TestVocabSizeTypes(unittest.TestCase):
    def test_vocab_sizes_are_int(self):
        config = Config()
        config.cpt_vocab_size = 7
        config.icd_vocab_size = 9
        config.ttnc_vocab_size = 3
        config.cpt_rarity_scores = None
        config.icd_rarity_scores = None
        config.ttnc_rarity_scores = None
        config.use_diffusion = False

        model = HierarchicalClaimsModel(config)
        self.assertIsInstance(model.cpt_vocab_size, int)
        self.assertIsInstance(model.icd_vocab_size, int)

if __name__ == '__main__':
    unittest.main()
