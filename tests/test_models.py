# tests/test_models.py
import unittest
import torch
from models.encoders import Level1Encoder, Level2Encoder
from models.prediction_blocks import Level1PredictionBlock, Level2PredictionBlock

class TestEncoders(unittest.TestCase):
    def test_level1_encoder(self):
        vocab_size = 100
        embedding_dim = 50
        encoder = Level1Encoder(vocab_size, embedding_dim, token_type='cpt')
        tokens = torch.randint(0, vocab_size, (2, 10))  # [batch_size, num_tokens]
        output = encoder(tokens, 'cpt')
        self.assertEqual(output.shape, (2, embedding_dim * 3))

    def test_level2_encoder(self):
        cpt_vocab_size = 100
        icd_vocab_size = 100
        ttnc_vocab_size = 50
        embedding_dim = 50
        encoder = Level2Encoder(cpt_vocab_size, icd_vocab_size, ttnc_vocab_size, embedding_dim)
        cpt_tokens = torch.randint(0, cpt_vocab_size, (2, 10))
        icd_tokens = torch.randint(0, icd_vocab_size, (2, 10))
        ttnc_tokens = torch.randint(0, ttnc_vocab_size, (2, 10))
        output = encoder(cpt_tokens, icd_tokens, ttnc_tokens)
        self.assertEqual(output.shape, (2, embedding_dim))

class TestPredictionBlocks(unittest.TestCase):
    def test_level2_prediction_block(self):
        embed_dim = 100
        output_dim = 50
        ttnc_vocab_size = 50
        max_seq_length = 100
        block = Level2PredictionBlock(embed_dim, output_dim, ttnc_vocab_size, max_seq_length)
        context_embeddings = torch.randn(2, 10, embed_dim)
        ttnc_tokens = torch.randint(0, ttnc_vocab_size, (2, 10))
        patient_representation, prediction = block(context_embeddings, ttnc_tokens)
        self.assertEqual(patient_representation.shape, (2, embed_dim))
        self.assertEqual(prediction.shape, (2, output_dim))

if __name__ == '__main__':
    unittest.main()
