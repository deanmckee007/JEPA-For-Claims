import sys
import unittest
from unittest import mock
import torch
from torch.utils.data import DataLoader, TensorDataset

import scripts.predict_to_csv as predict
from jepa_utils.config import Config


class TestPredictToCSV(unittest.TestCase):
    def test_main_writes_csv(self):
        cfg = Config()
        cfg.cpt_id_to_token = {0: '<PAD>', 1: 'cpt1'}
        cfg.icd_id_to_token = {0: '<PAD>', 1: 'icd1'}
        cfg.ttnc_id_to_token = {0: '<PAD>', 1: 'ttnc1'}
        cfg.cpt_vocab_size = 2
        cfg.icd_vocab_size = 2
        cfg.ttnc_vocab_size = 2

        cpt = torch.tensor([[[1, 0]]])
        icd = torch.tensor([[[1, 0]]])
        ttnc = torch.tensor([[1]])
        target = torch.tensor([0.0])
        dataset = TensorDataset(cpt, icd, ttnc, target)
        dataset.collate_fn = lambda batch: tuple(torch.stack(items) for items in zip(*batch))
        loader = DataLoader(dataset, batch_size=1)

        def fake_prepare(config):
            return dataset, loader, dataset, loader, cfg, dataset

        class DummyModel:
            def __init__(self):
                self.config = cfg
            def eval(self):
                pass
            def to(self, device):
                return self
            def __call__(self, cpt_tensor=None, icd_tensor=None, ttnc_tensor=None, generation=None, teacher_forcing=None):
                return {
                    'predicted_cpt_codes': cpt_tensor[:, -1],
                    'predicted_icd_codes': icd_tensor[:, -1],
                    'predicted_ttnc_code': ttnc_tensor[:, -1]
                }

        with mock.patch.object(predict, 'prepare_data', side_effect=fake_prepare):
            with mock.patch.object(predict.HierarchicalClaimsModel, 'load_from_checkpoint', return_value=DummyModel()):
                with mock.patch('pandas.DataFrame.to_csv') as mock_csv:
                    argv = ['predict_to_csv.py', 'model.ckpt']
                    with mock.patch.object(sys, 'argv', argv):
                        predict.main()
                    mock_csv.assert_called_once()


if __name__ == '__main__':
    unittest.main()
