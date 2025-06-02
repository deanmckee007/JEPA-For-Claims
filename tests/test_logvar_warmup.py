import unittest
import torch
from jepa_utils.config import Config
from jepa_models.hierarchical_model import HierarchicalClaimsModel

class TestLogvarWarmup(unittest.TestCase):
    def setUp(self):
        cfg = Config()
        cfg.use_diffusion = False
        cfg.use_sparse_autoencoder = False
        cfg.use_token_prediction_head = False
        cfg.use_predictor_head = False
        cfg.use_level1 = False
        cfg.cpt_rarity_scores = None
        cfg.icd_rarity_scores = None
        cfg.ttnc_rarity_scores = None
        cfg.cpt_vocab_size = 5
        cfg.icd_vocab_size = 5
        cfg.ttnc_vocab_size = 3
        cfg.max_cpt_tokens = 2
        cfg.max_icd_tokens = 2
        cfg.max_claims_len = 2
        cfg.warmup_logvar_epochs = 3
        self.model = HierarchicalClaimsModel(cfg)
        batch_size = 2
        self.batch = (
            torch.randint(1, cfg.cpt_vocab_size, (batch_size, cfg.max_claims_len, cfg.max_cpt_tokens)),
            torch.randint(1, cfg.icd_vocab_size, (batch_size, cfg.max_claims_len, cfg.max_icd_tokens)),
            torch.randint(1, cfg.ttnc_vocab_size, (batch_size, cfg.max_claims_len)),
            torch.randn(batch_size)
        )

    def test_warmup(self):
        optim, _ = self.model.configure_optimizers()
        optim = optim[0]
        grad_epoch0 = []
        grad_epoch4 = []
        from types import SimpleNamespace
        self.model.log = lambda *a, **k: None
        for epoch in range(5):
            self.model.trainer = SimpleNamespace(current_epoch=epoch, global_step=0)
            self.model.on_train_epoch_start()
            optim.zero_grad()
            loss = self.model.training_step(self.batch, 0)
            loss.backward()
            if epoch == 0:
                for p in self.model.log_vars.parameters():
                    grad_epoch0.append(p.grad)
            if epoch == 4:
                for p in self.model.log_vars.parameters():
                    grad_epoch4.append(p.grad)
            optim.step()
        for g in grad_epoch0:
            self.assertIsNone(g)
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in grad_epoch4))

if __name__ == '__main__':
    unittest.main()
