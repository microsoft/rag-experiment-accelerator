import unittest

from rag_experiment_accelerator.checkpoint.checkpoint_factory import CheckpointFactory
from rag_experiment_accelerator.checkpoint.null_checkpoint import NullCheckpoint
from rag_experiment_accelerator.config.config import ExecutionEnvironment


def dummy(word):
    return f"hello {word}"


class TestNullCheckpoint(unittest.TestCase):
    def test_wrapped_method_is_not_cached(self):
        checkpoint = CheckpointFactory.create_checkpoint(
            ExecutionEnvironment.LOCAL, False
        )
        self.assertIsInstance(checkpoint, NullCheckpoint)
        data_id = "unique_id"
        result1 = checkpoint.load_or_run(dummy, data_id, "first run")
        result2 = checkpoint.load_or_run(dummy, data_id, "second run")
        self.assertEqual(result1, "hello first run")
        self.assertEqual(result2, "hello second run")


if __name__ == "__main__":
    unittest.main()
