import unittest
from rag_experiment_accelerator.checkpoint.null_checkpoint import (
    NullCheckpoint,
)


class TestNullCheckpoint(unittest.TestCase):
    def test_save_does_not_do_anything(self):
        checkpoint = NullCheckpoint()
        checkpoint.save("test_data")
        self.assertIsNone(checkpoint.load())

    def test_exists(self):
        checkpoint = NullCheckpoint()
        self.assertEqual(checkpoint.exists(), False)
        checkpoint.save("test_data")
        self.assertEqual(checkpoint.exists(), False)


if __name__ == "__main__":
    unittest.main()
