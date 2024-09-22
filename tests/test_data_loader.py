# tests/test_data_loader.py
import unittest
from data.data_loader import load_data

class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        training_path = 'path/to/training_set.pckl'
        testing_path = 'path/to/test_set.pckl'
        pd_training_df, pd_test_df = load_data(training_path, testing_path)
        self.assertIsInstance(pd_training_df, pd.DataFrame)
        self.assertIsInstance(pd_test_df, pd.DataFrame)
        # Add more assertions as needed

if __name__ == '__main__':
    unittest.main()
