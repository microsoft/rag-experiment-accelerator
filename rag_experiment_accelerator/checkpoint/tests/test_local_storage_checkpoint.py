import unittest
import os
from unittest.mock import MagicMock
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)
import tempfile
import shutil


class TestLocalStorageCheckpoint(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.mkdtemp()
        self.checkpoints_dir = f"{temp_dir}/checkpoints"
        self.config = MagicMock()
        self.config.artifacts_dir = temp_dir

    def tearDown(self):
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir)

    def test_save_load(self):
        checkpoint = LocalStorageCheckpoint(
            "test_save_load", "test_config", self.config
        )
        checkpoint.save("test_data")
        self.assertEqual(checkpoint.load(), ["test_data"])

    def test_exists(self):
        checkpoint = LocalStorageCheckpoint("test_exists", "test_config", self.config)
        self.assertEqual(checkpoint.exists(), False)
        checkpoint.save("test_data")
        self.assertEqual(checkpoint.exists(), True)


if __name__ == "__main__":
    unittest.main()
