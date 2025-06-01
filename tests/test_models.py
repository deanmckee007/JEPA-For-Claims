# tests/test_models.py
import unittest
import torch
from jepa_models.encoders import Level1Encoder, Level2Encoder
from jepa_models.prediction_blocks import Level1PredictionBlock, Level2PredictionBlock

class TestEncoders(unittest.TestCase):
    def test_level1_encoder(self):
        cpt_vocab_size = 100
        icd_vocab_size = 80
        embedding_dim = 50
        encoder = Level1Encoder(cpt_vocab_size, icd_vocab_size, embedding_dim)
        tokens = torch.randint(0, cpt_vocab_size, (2, 3, 10))  # [batch_size, num_claims, num_tokens]
        output, mask = encoder(tokens, 'cpt')
        self.assertEqual(output.shape, (2, 3, embedding_dim * 3))

    def test_level2_encoder(self):
        cpt_vocab_size = 100
        icd_vocab_size = 100
        ttnc_vocab_size = 50
        embedding_dim = 50
        encoder = Level2Encoder(cpt_vocab_size, icd_vocab_size, ttnc_vocab_size, embedding_dim)
        cpt_tokens = torch.randint(0, cpt_vocab_size, (2, 4, 10))
        icd_tokens = torch.randint(0, icd_vocab_size, (2, 4, 8))
        ttnc_tokens = torch.randint(0, ttnc_vocab_size, (2, 4))
        output = encoder(cpt_tokens, icd_tokens, ttnc_tokens)
        self.assertEqual(output.shape, (2, 4, embedding_dim))

class TestPredictionBlocks(unittest.TestCase):
    def test_level2_prediction_block(self):
        embed_dim = 100
        output_dim = 50
        cpt_vocab_size = 200
        icd_vocab_size = 150
        ttnc_vocab_size = 50
        max_seq_length = 100
        block = Level2PredictionBlock(embed_dim, output_dim, cpt_vocab_size, icd_vocab_size, ttnc_vocab_size, max_seq_length)
        context_embeddings = torch.randn(2, 10, embed_dim)
        ttnc_tokens = torch.randint(0, ttnc_vocab_size, (2, 10))
        patient_representation, prediction = block(context_embeddings, ttnc_tokens)
        self.assertEqual(patient_representation.shape, (2, embed_dim))
        self.assertEqual(prediction.shape, (2, output_dim))

    def test_level2_prediction_block_context_pooled(self):
        embed_dim = 100
        output_dim = 50
        cpt_vocab_size = 200
        icd_vocab_size = 150
        ttnc_vocab_size = 50
        max_seq_length = 100
        block = Level2PredictionBlock(
            embed_dim,
            output_dim,
            cpt_vocab_size,
            icd_vocab_size,
            ttnc_vocab_size,
            max_seq_length,
            use_context_pooled_patient_representation=True,
        )
        context_embeddings = torch.randn(2, 10, embed_dim)
        ttnc_tokens = torch.randint(0, ttnc_vocab_size, (2, 10))
        patient_representation, prediction = block(context_embeddings, ttnc_tokens)
        self.assertEqual(patient_representation.shape, (2, embed_dim * 2))
        self.assertEqual(prediction.shape, (2, output_dim))

    def test_level2_prediction_block_zero_vocab(self):
        """Ensure initialization succeeds when vocab sizes are zero."""
        embed_dim = 8
        output_dim = 4
        block = Level2PredictionBlock(
            embed_dim,
            output_dim,
            cpt_vocab_size=0,
            icd_vocab_size=0,
            ttnc_vocab_size=0,
            max_seq_length=10,
        )
        context_embeddings = torch.randn(1, 3, embed_dim)
        ttnc_tokens = torch.zeros(1, 3, dtype=torch.long)
        patient_rep, pred = block(context_embeddings, ttnc_tokens)
        self.assertEqual(patient_rep.shape, (1, embed_dim))
        self.assertEqual(pred.shape, (1, output_dim))


class TestHierarchicalModelVicreg(unittest.TestCase):
    def test_stage1_uses_vicreg_level2(self):
        from jepa_models.hierarchical_model import HierarchicalClaimsModel
        from jepa_utils.config import Config

        config = Config()
        config.use_diffusion = False
        config.use_sparse_autoencoder = False
        config.use_token_prediction_head = False
        config.use_predictor_head = False
        config.use_level1 = False
        config.cpt_rarity_scores = None
        config.icd_rarity_scores = None
        config.ttnc_rarity_scores = None
        config.cpt_vocab_size = 10
        config.icd_vocab_size = 10
        config.ttnc_vocab_size = 5
        config.embedding_dim = 4
        config.output_dim = config.embedding_dim
        config.max_cpt_tokens = 3
        config.max_icd_tokens = 3
        config.max_claims_len = 4

        model = HierarchicalClaimsModel(config)
        batch_size = 2
        cpt_tensor = torch.randint(1, config.cpt_vocab_size, (batch_size, config.max_claims_len, config.max_cpt_tokens))
        icd_tensor = torch.randint(1, config.icd_vocab_size, (batch_size, config.max_claims_len, config.max_icd_tokens))
        ttnc_tensor = torch.randint(1, config.ttnc_vocab_size, (batch_size, config.max_claims_len))
        target = torch.randn(batch_size)

        outputs = model.training_forward(cpt_tensor, icd_tensor, ttnc_tensor, target)
        self.assertGreater(outputs['vicreg_loss_lvl2'].item(), 0)

if __name__ == '__main__':
    unittest.main()
