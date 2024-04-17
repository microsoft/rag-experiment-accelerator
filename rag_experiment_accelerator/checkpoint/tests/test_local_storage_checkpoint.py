import unittest
import os
from unittest.mock import MagicMock
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)
import tempfile
import shutil


def dummy(word):
    return f"hello {word}"


class TestLocalStorageCheckpoint(unittest.TestCase):
    def setUp(self):
        temp_dir = tempfile.mkdtemp()
        self.checkpoints_dir = f"{temp_dir}/checkpoints"
        self.config = MagicMock()
        self.config.artifacts_dir = temp_dir

    def tearDown(self):
        if os.path.exists(self.checkpoints_dir):
            shutil.rmtree(self.checkpoints_dir)

    def test_wrapped_method_is_cached(self):
        checkpoint = LocalStorageCheckpoint(
            "test_save_load", "test_config", self.config
        )
        data_id = "unique_id"
        result1 = checkpoint.load_or_run(dummy, data_id, "first run")
        result2 = checkpoint.load_or_run(dummy, data_id, "second run")
        self.assertEqual(result1, "hello first run")
        self.assertEqual(result2, "hello first run")

    def test_ids_are_saved(self):
        checkpoint = LocalStorageCheckpoint("test_exists", "test_config", self.config)
        checkpoint.load_or_run(dummy, "id1", "one")
        checkpoint.load_or_run(dummy, "id1", "two")
        checkpoint.load_or_run(dummy, "id2", "three")
        checkpoint.load_or_run(dummy, "id2", "four")
        checkpoint.load_or_run(dummy, "id3", "five")

        checkpoint_ids = checkpoint.get_saved_ids(dummy)
        self.assertEqual(checkpoint_ids, set(["id1", "id2", "id3"]))


if __name__ == "__main__":
    unittest.main()
