import unittest
from unittest.mock import patch, MagicMock
import torch
from torch.utils.data import DataLoader, TensorDataset
from jepa_utils.config import Config
import scripts.train as train

class TestPretrainDiffusionFlag(unittest.TestCase):
    def setUp(self):
        dataset = TensorDataset(torch.tensor([1]))
        self.loader = DataLoader(dataset)

    @patch('scripts.train.os.path.exists')
    @patch('scripts.train.pl.Trainer')
    @patch('scripts.train.prepare_data')
    @patch('scripts.train.load_claims_model_checkpoint')
    @patch('scripts.train.HierarchicalClaimsModel')
    @patch('scripts.train.DiffusionModel')
    @patch('scripts.train.Config')
    def test_pretrain_enabled(self, mock_config, mock_diffusion, mock_hier, mock_load, mock_prepare, mock_trainer, mock_exists):
        cfg = Config()
        cfg.use_diffusion = True
        cfg.pretrain_diffusion = True
        cfg.use_generative_save = False
        cfg.use_token_prediction_head = False
        cfg.use_lr_find = False
        cfg.use_plotting = False
        cfg.representation_pretrain_epochs = 0
        cfg.generator_train_epochs = 1
        cfg.joint_train_epochs = 0
        mock_config.return_value = cfg
        mock_prepare.return_value = (None, self.loader, None, None, cfg, None)
        mock_exists.return_value = True
        trainer_instance1 = MagicMock()
        trainer_instance2 = MagicMock()
        mock_trainer.side_effect = [trainer_instance1, trainer_instance2]

        train.main()

        self.assertEqual(mock_trainer.call_count, 2)
        trainer_instance1.fit.assert_called_once()

    @patch('scripts.train.os.path.exists')
    @patch('scripts.train.pl.Trainer')
    @patch('scripts.train.prepare_data')
    @patch('scripts.train.load_claims_model_checkpoint')
    @patch('scripts.train.HierarchicalClaimsModel')
    @patch('scripts.train.DiffusionModel')
    @patch('scripts.train.Config')
    def test_pretrain_disabled(self, mock_config, mock_diffusion, mock_hier, mock_load, mock_prepare, mock_trainer, mock_exists):
        cfg = Config()
        cfg.use_diffusion = True
        cfg.pretrain_diffusion = False
        cfg.use_generative_save = False
        cfg.use_token_prediction_head = False
        cfg.use_lr_find = False
        cfg.use_plotting = False
        cfg.representation_pretrain_epochs = 0
        cfg.generator_train_epochs = 1
        cfg.joint_train_epochs = 0
        mock_config.return_value = cfg
        mock_prepare.return_value = (None, self.loader, None, None, cfg, None)
        mock_exists.return_value = True
        trainer_instance = MagicMock()
        mock_trainer.return_value = trainer_instance

        train.main()

        # Only main training should run
        self.assertEqual(mock_trainer.call_count, 1)
        trainer_instance.fit.assert_called_once()

if __name__ == '__main__':
    unittest.main()
