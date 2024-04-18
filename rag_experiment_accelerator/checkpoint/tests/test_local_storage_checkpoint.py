import unittest
import os
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)
import tempfile
import shutil


def dummy(word):
    return f"hello {word}"


class TestLocalStorageCheckpoint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_wrapped_method_is_cached(self):
        checkpoint = LocalStorageCheckpoint(
            checkpoint_name="test_save_load", directory=self.temp_dir
        )
        data_id = "unique_id"
        result1 = checkpoint.load_or_run(dummy, data_id, "first run")
        result2 = checkpoint.load_or_run(dummy, data_id, "second run")
        self.assertEqual(result1, "hello first run")
        self.assertEqual(result2, "hello first run")


if __name__ == "__main__":
    unittest.main()
