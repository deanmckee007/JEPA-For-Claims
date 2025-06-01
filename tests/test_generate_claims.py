import os
import sys
import tempfile
import torch
import types
import unittest
from unittest import mock

import scripts.generate_claims as gen
from jepa_utils.config import Config
from jepa_models.hierarchical_model import HierarchicalClaimsModel

class TestGenerateClaimsLoad(unittest.TestCase):
    def test_load_vocab_from_checkpoint(self):
        # Build a minimal state_dict with embedding weights
        state_dict = {
            'context_encoder_lvl2.cpt_embedding.weight': torch.randn(5, 4),
            'context_encoder_lvl2.icd_embedding.weight': torch.randn(6, 4),
            'context_encoder_lvl2.ttnc_embedding.weight': torch.randn(3, 4),
        }
        ckpt = {'state_dict': state_dict, 'hyper_parameters': {'config': {}}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.ckpt')
            torch.save(ckpt, path)

            captured = {}
            def fake_load(path_arg, config, strict=False):
                captured['config'] = config
                class Dummy:
                    def __init__(self):
                        self.config = config
                        self.diffusion_model = self
                        self.condition_dim = config.output_dim
                    def eval(self):
                        pass
                    def to(self, device):
                        return self
                    def generate_claim(self, condition, seq_len):
                        return torch.zeros(condition.size(0), seq_len, dtype=torch.long)
                return Dummy()

            with mock.patch.object(HierarchicalClaimsModel, 'load_from_checkpoint', side_effect=fake_load):
                with mock.patch.object(torch.cuda, 'is_available', return_value=False):
                    argv = ['generate_claims.py', path, '--num', '1']
                    with mock.patch.object(sys, 'argv', argv):
                        gen.main()
            cfg = captured['config']
            self.assertEqual(cfg.cpt_vocab_size, 5)
            self.assertEqual(cfg.icd_vocab_size, 6)
            self.assertEqual(cfg.ttnc_vocab_size, 3)

if __name__ == '__main__':
    unittest.main()
