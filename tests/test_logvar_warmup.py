import unittest
import torch
from jepa_utils.config import Config
from jepa_models.hierarchical_model import HierarchicalClaimsModel

class TestLogVarWarmup(unittest.TestCase):
    def test_logvar_freeze_warmup(self):
        cfg = Config()
        cfg.cpt_vocab_size = 3
        cfg.icd_vocab_size = 3
        cfg.ttnc_vocab_size = 2
        cfg.cpt_rarity_scores = None
        cfg.icd_rarity_scores = None
        cfg.ttnc_rarity_scores = None
        cfg.use_diffusion = False
        cfg.use_sparse_autoencoder = False
        cfg.use_token_prediction_head = False
        cfg.use_predictor_head = False
        cfg.use_level1 = False
        cfg.warmup_logvar_epochs = 3

        model = HierarchicalClaimsModel(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        batch_size = 2
        cpt = torch.randint(1, cfg.cpt_vocab_size, (batch_size, cfg.max_claims_len, cfg.max_cpt_tokens))
        icd = torch.randint(1, cfg.icd_vocab_size, (batch_size, cfg.max_claims_len, cfg.max_icd_tokens))
        ttnc = torch.randint(1, cfg.ttnc_vocab_size, (batch_size, cfg.max_claims_len))
        target = torch.randn(batch_size)
        batch = (cpt, icd, ttnc, target)

        import types
        for epoch in range(5):
            model._trainer = types.SimpleNamespace(current_epoch=epoch, global_step=0)
            model.log = lambda *a, **k: None
            model.on_train_epoch_start()
            optimizer.zero_grad()
            loss = model.training_step(batch, 0)
            loss.backward()
            optimizer.step()
            grads = [p.grad for p in model.log_vars.values()]
            if epoch == 0:
                self.assertTrue(all(g is None or torch.all(g == 0) for g in grads))
            if epoch == 4:
                self.assertTrue(any(g is not None and torch.any(g != 0) for g in grads))

if __name__ == '__main__':
    unittest.main()
