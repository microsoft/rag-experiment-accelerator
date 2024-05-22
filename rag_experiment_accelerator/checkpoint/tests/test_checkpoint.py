from unittest.mock import MagicMock
import pytest
from unittest.mock import patch
from rag_experiment_accelerator.checkpoint.checkpoint_factory import (
    get_checkpoint_instance,
    create_checkpoint,
)
from rag_experiment_accelerator.checkpoint.local_storage_checkpoint import (
    LocalStorageCheckpoint,
)
from rag_experiment_accelerator.checkpoint.null_checkpoint import NullCheckpoint
from rag_experiment_accelerator.config.config import ExecutionEnvironment


@pytest.fixture
def mock_checkpoints():
    with patch.object(
        LocalStorageCheckpoint, "__init__", return_value=None
    ), patch.object(NullCheckpoint, "__init__", return_value=None):
        yield


def test_get_checkpoint_without_init_fails():
    with pytest.raises(Exception) as e_info:
        get_checkpoint_instance()
    assert (
        str(e_info.value)
        == "Checkpoint not initialized yet. Call CheckpointFactory.create_checkpoint() first."
    )


def test_get_checkpoint_for_local_executions(mock_checkpoints):
    config = MagicMock()
    config.execution_environment = ExecutionEnvironment.LOCAL
    config.use_checkpoints = True

    checkpoint = create_checkpoint(config.execution_environment, config.use_checkpoints)
    assert isinstance(checkpoint, LocalStorageCheckpoint)


def test_get_checkpoint_for_azure_ml(mock_checkpoints):
    config = MagicMock()
    config.execution_environment = ExecutionEnvironment.AZURE_ML
    config.use_checkpoints = True

    checkpoint = create_checkpoint(config.execution_environment, config.use_checkpoints)

    # currently not supposed for Azure ML, so it should return NullCheckpoint
    assert isinstance(checkpoint, NullCheckpoint)


def test_get_checkpoint_when_should_not_use_checkpoints_locally(mock_checkpoints):
    config = MagicMock()
    config.execution_environment = ExecutionEnvironment.LOCAL
    config.use_checkpoints = False

    checkpoint = create_checkpoint(config.execution_environment, config.use_checkpoints)

    assert isinstance(checkpoint, NullCheckpoint)


def test_get_checkpoint_when_should_not_use_checkpoints_in_azure_ml(mock_checkpoints):
    config = MagicMock()
    config.execution_environment = ExecutionEnvironment.AZURE_ML
    config.use_checkpoints = False

    checkpoint = create_checkpoint(config.execution_environment, config.use_checkpoints)

    assert isinstance(checkpoint, NullCheckpoint)
