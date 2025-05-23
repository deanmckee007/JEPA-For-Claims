import unittest
from jepa_utils.config import Config
from jepa_models.hierarchical_model import HierarchicalClaimsModel

class TestFreezeUnfreeze(unittest.TestCase):
    def test_freeze_and_unfreeze(self):
        cfg = Config()
        cfg.cpt_vocab_size = 5
        cfg.icd_vocab_size = 5
        cfg.ttnc_vocab_size = 3
        cfg.cpt_rarity_scores = None
        cfg.icd_rarity_scores = None
        cfg.ttnc_rarity_scores = None
        cfg.use_diffusion = False
        model = HierarchicalClaimsModel(cfg)
        model.freeze_encoders(1)
        frozen = [p.requires_grad for p in model.context_encoder_lvl2.parameters()]
        self.assertTrue(any(not f for f in frozen))
        model.unfreeze_encoders()
        self.assertTrue(all(p.requires_grad for p in model.context_encoder_lvl2.parameters()))

if __name__ == '__main__':
    unittest.main()
