import os
import tempfile
import torch
import unittest

from jepa_utils.config import Config
from jepa_models.hierarchical_model import HierarchicalClaimsModel
from jepa_utils.checkpoint_utils import checkpoint_has_prefixed_keys, freeze_non_diffusion


class TestLoadDiffusionFromJoint(unittest.TestCase):
    def test_freeze_non_diffusion_from_joint_ckpt(self):
        cfg = Config()
        cfg.cpt_vocab_size = 3
        cfg.icd_vocab_size = 3
        cfg.ttnc_vocab_size = 2
        cfg.cpt_rarity_scores = None
        cfg.icd_rarity_scores = None
        cfg.ttnc_rarity_scores = None
        cfg.use_diffusion = True

        model = HierarchicalClaimsModel(cfg)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "joint.ckpt")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hyper_parameters": {"config": {}},
                    "pytorch-lightning_version": "2.2.0",
                },
                path,
            )

            self.assertTrue(checkpoint_has_prefixed_keys(path))

            loaded = HierarchicalClaimsModel.load_from_checkpoint(path, config=cfg)
            freeze_non_diffusion(loaded)
            for name, param in loaded.named_parameters():
                if not name.startswith("diffusion_model"):
                    self.assertFalse(param.requires_grad)


if __name__ == "__main__":
    unittest.main()
