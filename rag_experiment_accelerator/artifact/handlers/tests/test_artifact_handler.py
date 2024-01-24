from unittest.mock import Mock
import pytest

from rag_experiment_accelerator.artifact.handlers.artifact_handler import (
    ArtifactHandler,
)
from rag_experiment_accelerator.artifact.handlers.exceptions import LoadException


def test_loads():
    data = "This is test data"
    mock_writer = Mock()
    mock_loader = Mock()
    mock_loader.can_handle.return_value = True
    mock_loader.load.return_value = [data]

    handler = ArtifactHandler("data_location", writer=mock_writer, loader=mock_loader)

    name = "test.jsonl"
    loaded_data = handler.load(name)

    assert loaded_data == [data]


def test_save_dict():
    mock_writer = Mock()
    mock_loader = Mock()

    handler = ArtifactHandler("data_location", writer=mock_writer, loader=mock_loader)

    dict_to_save = {"testing": 123, "mic": "check"}
    artifact_name = "test.jsonl"
    handler.save_dict(dict_to_save, "test.jsonl")
    path = f"{handler.data_location}/{artifact_name}"

    assert mock_writer.write.call_count == 1
    assert mock_writer.write.called_with(dict_to_save, path)


def test_loads_raises_no_data_returned():
    mock_writer = Mock()
    mock_loader = Mock()
    mock_loader.can_handle.return_value = True
    mock_loader.load.return_value = []
    handler = ArtifactHandler("data_location", writer=mock_writer, loader=mock_loader)
    name = "test.jsonl"

    with pytest.raises(LoadException):
        handler.load(name)


def test_load_raises_cant_handle():
    mock_writer = Mock()
    mock_loader = Mock()
    handler = ArtifactHandler("data_location", writer=mock_writer, loader=mock_loader)

    mock_loader.can_handle.return_value = False

    with pytest.raises(LoadException):
        handler.load("test.txt")


def test_handle_archive():
    mock_writer = Mock()
    mock_loader = Mock()
    mock_writer.exists.return_value = True
    data_location = "data_location"
    handler = ArtifactHandler(data_location, writer=mock_writer, loader=mock_loader)

    name = "test.jsonl"
    dest = handler.handle_archive(name)

    src = f"{data_location}/{name}"
    mock_writer.copy.assert_called_once_with(src, dest)
    mock_writer.delete.assert_called_once_with(src)


def test_handle_archive_no_op():
    mock_writer = Mock()
    mock_loader = Mock()
    # only archive is exists
    mock_writer.exists.return_value = False
    handler = ArtifactHandler("data_location", writer=mock_writer, loader=mock_loader)

    dest = handler.handle_archive("test.jsonl")

    mock_writer.copy.assert_not_called()
    mock_writer.delete.assert_not_called()
    assert dest is None
