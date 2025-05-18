# tests/test_data_loader.py
import os
import tempfile
import unittest
import pandas as pd

from data.data_loader import load_data


class TestDataLoader(unittest.TestCase):
    def test_load_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.pckl")
            test_path = os.path.join(tmpdir, "test.pckl")
            # create dummy dataframes
            train_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
            test_df = pd.DataFrame({"c": [5, 6], "d": [7, 8]})
            train_df.to_pickle(train_path)
            test_df.to_pickle(test_path)

            loaded_train, loaded_test = load_data(train_path, test_path)
            self.assertTrue(loaded_train.equals(train_df))
            self.assertTrue(loaded_test.equals(test_df))

    def test_load_data_pkl_extension(self):
        """Ensure loading works with ``.pkl`` files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.pkl")
            test_path = os.path.join(tmpdir, "test.pkl")
            train_df = pd.DataFrame({"a": [9, 10]})
            test_df = pd.DataFrame({"b": [11, 12]})
            train_df.to_pickle(train_path)
            test_df.to_pickle(test_path)

            loaded_train, loaded_test = load_data(train_path, test_path)
            self.assertTrue(loaded_train.equals(train_df))
            self.assertTrue(loaded_test.equals(test_df))


if __name__ == "__main__":
    unittest.main()
