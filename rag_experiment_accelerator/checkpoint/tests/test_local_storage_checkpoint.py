import unittest
import os
import tempfile
import shutil

from rag_experiment_accelerator.checkpoint.checkpoint_decorator import (
    cache_with_checkpoint,
)
from rag_experiment_accelerator.checkpoint.checkpoint_factory import (
    create_checkpoint,
    reset_checkpoint_instance,
)
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)


@cache_with_checkpoint(key="call_identifier")
def dummy(word, call_identifier):
    return f"hello {word}"


class TestLocalStorageCheckpoint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.checkpoint = create_checkpoint(
            type="local", enable_checkpoints=True, checkpoints_directory=self.temp_dir
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        reset_checkpoint_instance()

    def test_wrapped_method_is_cached(self):
        assert isinstance(self.checkpoint, LocalStorageCheckpoint)

        data_id = "same_id"
        result1 = dummy("first run", data_id)
        result2 = dummy("second run", data_id)
        self.assertEqual(result1, "hello first run")
        self.assertEqual(result2, "hello first run")


if __name__ == "__main__":
    unittest.main()
