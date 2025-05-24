import unittest
import torch
from jepa_utils.prediction_utils import decode_predicted_codes

class TestDecodePredictedCodes(unittest.TestCase):
    def test_decode_multi_hot(self):
        tensor = torch.tensor([[0,1,0,1]])
        id_to_token = {0:'<PAD>',1:'A',2:'B',3:'C'}
        decoded = decode_predicted_codes(tensor, id_to_token)
        self.assertEqual(decoded, [['A','C']])

    def test_decode_token_ids(self):
        tensor = torch.tensor([[2,3,0]])
        id_to_token = {0:'<PAD>',2:'B',3:'C'}
        decoded = decode_predicted_codes(tensor, id_to_token)
        self.assertEqual(decoded, [['B','C']])

    def test_decode_single_token(self):
        tensor = torch.tensor([3])
        id_to_token = {3:'C'}
        decoded = decode_predicted_codes(tensor, id_to_token)
        self.assertEqual(decoded, [['C']])

if __name__ == '__main__':
    unittest.main()
