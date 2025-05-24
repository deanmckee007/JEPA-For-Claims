import unittest
import torch
from jepa_models.sparse_autoencoder import TopKActivation

class TestTopKActivation(unittest.TestCase):
    def test_topk_indices(self):
        torch.manual_seed(0)
        x = torch.tensor([[1.0, -2.0, 0.5, 4.0, -3.0]])
        activation = TopKActivation(k=3)
        out = activation(x)
        # Expected indices of the top 3 absolute values
        _, expected_idx = torch.topk(x.abs(), 3, dim=1)
        nonzero_idx = out.nonzero(as_tuple=False)[:, 1]
        self.assertEqual(set(expected_idx.view(-1).tolist()), set(nonzero_idx.tolist()))
        self.assertTrue(torch.all(out[:, nonzero_idx] != 0))
        self.assertTrue((out == 0).sum().item() == x.numel() - 3)

if __name__ == '__main__':
    unittest.main()
