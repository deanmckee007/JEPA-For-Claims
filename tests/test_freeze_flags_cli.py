import unittest
from unittest.mock import patch
import scripts.train as train

class TestFreezeFlags(unittest.TestCase):
    def test_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            train.main(["--freeze_diffusion", "--freeze_jepa"])

if __name__ == "__main__":
    unittest.main()
