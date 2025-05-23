import os
import unittest
from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import DataLoader, TensorDataset
from jepa_utils.config import Config
import scripts.train as train


class TestAutoLoadEncoder(unittest.TestCase):
    def setUp(self):
        dataset = TensorDataset(torch.tensor([1]))
        self.loader = DataLoader(dataset)

    @patch('scripts.train.pl.Trainer')
    @patch('scripts.train.prepare_data')
    @patch('scripts.train.HierarchicalClaimsModel')
    @patch('scripts.train.Config')
    def test_auto_load_latest_checkpoint(self, mock_config, mock_model, mock_prepare, mock_trainer):
        cfg = Config()
        cfg.representation_pretrain_epochs = 0
        cfg.generator_train_epochs = 1
        cfg.use_generative_save = False
        cfg.use_token_prediction_head = False
        cfg.use_diffusion = False
        cfg.use_lr_find = False
        cfg.use_plotting = False
        cfg.pretrained_encoder_ckpt = ""
        mock_config.return_value = cfg
        mock_prepare.return_value = (None, self.loader, None, None, cfg, None)
        trainer_instance = MagicMock()
        mock_trainer.return_value = trainer_instance

        os.makedirs('checkpoints', exist_ok=True)
        dummy_ckpt = os.path.join('checkpoints', 'encoder_only_dummy.ckpt')
        with open(dummy_ckpt, 'w') as f:
            f.write('x')

        model_instance = MagicMock()
        mock_model.load_from_checkpoint.return_value = model_instance

        train.main()

        self.assertEqual(
            mock_model.load_from_checkpoint.call_args[0][0], dummy_ckpt
        )
        self.assertTrue(mock_model.load_from_checkpoint.called)
        model_instance.freeze_encoder.assert_called_once()


if __name__ == '__main__':
    unittest.main()
