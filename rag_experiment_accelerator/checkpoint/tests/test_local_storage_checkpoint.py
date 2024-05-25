import unittest
import os
import tempfile
import shutil
from unittest.mock import MagicMock

from rag_experiment_accelerator.checkpoint.checkpoint_factory import (
    get_checkpoint,
    init_checkpoint,
)
from rag_experiment_accelerator.checkpoint.checkpoint_decorator import (
    cache_with_checkpoint,
)
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)


@cache_with_checkpoint(id="call_identifier")
def dummy(word, call_identifier):
    return f"hello {word}"


class TestLocalStorageCheckpoint(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_wrapped_method_is_cached(self):
        config = MagicMock()
        config.use_checkpoints = True
        config.artifacts_dir = self.temp_dir
        init_checkpoint(config)
        checkpoint = get_checkpoint()
        assert isinstance(checkpoint, LocalStorageCheckpoint)

        data_id = "same_id"
        result1 = dummy("first run", data_id)
        result2 = dummy("second run", data_id)
        self.assertEqual(result1, "hello first run")
        self.assertEqual(result2, "hello first run")


if __name__ == "__main__":
    unittest.main()
