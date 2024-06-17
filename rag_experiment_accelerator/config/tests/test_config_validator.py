import pytest
import os

import requests
from rag_experiment_accelerator.config.config_validator import (
    fetch_json_schema,
    fetch_json_schema_from_url,
    fetch_json_schema_from_file,
    get_normalised_schema_path,
)
from unittest.mock import MagicMock, mock_open, patch


@patch("requests.get")
def test_fetch_json_schema_from_url_returns_json(mock_get):
    schema_url = "http://test.com/schema.json"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"key": "value"}

    mock_get.return_value = mock_response

    result = fetch_json_schema_from_url(schema_url)
    assert result == mock_response.json.return_value


@patch("requests.get")
def test_fetch_json_schema_from_url_raises_error_for_timeout(mock_get):
    schema_url = "http://test.com/schema.json"

    mock_get.side_effect = requests.exceptions.Timeout

    with pytest.raises(requests.exceptions.Timeout):
        fetch_json_schema_from_url(schema_url)


@patch("os.path.isfile")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"$schema": "http://test.com/schema.json"}',
)
def test_fetch_json_schema_from_file_returns_json_from_file(mock_open, mock_isfile):
    mock_isfile.return_value = True

    cwd = os.getcwd()
    schema_file_path = os.path.join(cwd, "example.schema.json")
    source_file_path = os.path.join(cwd, "source_file.json")

    result = fetch_json_schema_from_file(schema_file_path, source_file_path)

    mock_open.assert_called_once_with(schema_file_path, "r", encoding="utf8")
    assert result == {"$schema": "http://test.com/schema.json"}


@patch("os.path.isfile")
@patch(
    "builtins.open",
    new_callable=mock_open,
    read_data='{"$schema": "http://test.com/schema.json"}',
)
def test_fetch_json_schema_from_file_uses_a_relative_path(mock_open, mock_isfile):
    mock_isfile.return_value = True

    schema_file_path = "../../config.schema.json"
    source_file_path = "/home/runner/work/rag-experiment-accelerator/rag-experiment-accelerator/.github/workflows/config.json"

    fetch_json_schema_from_file(schema_file_path, source_file_path)

    mock_open.assert_called_once_with(
        "/home/runner/work/rag-experiment-accelerator/rag-experiment-accelerator/config.schema.json",
        "r",
        encoding="utf8",
    )


@patch("os.path.isfile")
def test_fetch_json_schema_from_file_raises_error_for_non_file_input(mock_isfile):
    mock_isfile.return_value = False

    cwd = os.getcwd()
    schema_file_path = os.path.join(cwd, "not_a_file")
    source_file_path = os.path.join(cwd, "source_file.json")

    with pytest.raises(ValueError):
        fetch_json_schema_from_file(schema_file_path, source_file_path)


def test_get_normalised_schema_path():
    schema_file_path = "../../config.schema.json"
    source_file_path = "/home/runner/work/rag-experiment-accelerator/rag-experiment-accelerator/.github/workflows/config.json"

    result = get_normalised_schema_path(schema_file_path, source_file_path)
    assert (
        result
        == "/home/runner/work/rag-experiment-accelerator/rag-experiment-accelerator/config.schema.json"
    )


@patch("rag_experiment_accelerator.config.config_validator.schema_cache")
def test_fetch_json_schema_returns_from_cache(mock_schema_cache):
    schema_reference = "http://test.com/schema.json"
    schema_cache = {schema_reference: {"key": "value"}}

    mock_schema_cache.__contains__.return_value = schema_cache.__contains__
    mock_schema_cache.__getitem__.side_effect = schema_cache.__getitem__

    result = fetch_json_schema(schema_reference, "source_file.json")
    assert result == schema_cache[schema_reference]


@patch("rag_experiment_accelerator.config.config_validator.schema_cache")
@patch("rag_experiment_accelerator.config.config_validator.fetch_json_schema_from_url")
def test_fetch_json_schema_updates_cache(
    mock_fetch_json_schema_from_url, mock_schema_cache
):
    schema_reference = "http://test.com/schema.json"
    schema_cache = {}

    mock_schema_cache.__contains__.return_value = False
    mock_schema_cache.__setitem__.side_effect = schema_cache.__setitem__
    mock_fetch_json_schema_from_url.return_value = {
        "$schema": "http://test.com/schema.json"
    }

    fetch_json_schema(schema_reference, "source_file.json")
    assert schema_cache == {
        schema_reference: {"$schema": "http://test.com/schema.json"}
    }


@patch("rag_experiment_accelerator.config.config_validator.schema_cache")
@patch("rag_experiment_accelerator.config.config_validator.fetch_json_schema_from_url")
def test_fetch_json_schema_returns_from_url_when_http(
    mock_fetch_json_schema_from_url, mock_schema_cache
):
    schema_reference = "http://test.com/schema.json"

    mock_schema_cache.__contains__.return_value = False
    mock_fetch_json_schema_from_url.return_value = {
        "$schema": "http://test.com/schema.json"
    }

    fetch_json_schema(schema_reference, "source_file.json")
    mock_fetch_json_schema_from_url.assert_called_once_with(schema_reference)


@patch("rag_experiment_accelerator.config.config_validator.schema_cache")
@patch("rag_experiment_accelerator.config.config_validator.fetch_json_schema_from_url")
def test_fetch_json_schema_returns_from_url_when_https(
    mock_fetch_json_schema_from_url, mock_schema_cache
):
    schema_reference = "https://test.com/schema.json"

    mock_schema_cache.__contains__.return_value = False
    mock_fetch_json_schema_from_url.return_value = {
        "$schema": "http://test.com/schema.json"
    }

    fetch_json_schema(schema_reference, "source_file.json")
    mock_fetch_json_schema_from_url.assert_called_once_with(schema_reference)


@patch("rag_experiment_accelerator.config.config_validator.schema_cache")
@patch("rag_experiment_accelerator.config.config_validator.fetch_json_schema_from_file")
def test_fetch_json_schema_returns_from_file(
    mock_fetch_json_schema_from_file, mock_schema_cache
):
    schema_reference = "./schema.json"

    mock_schema_cache.__contains__.return_value = False
    mock_fetch_json_schema_from_file.return_value = {
        "$schema": "http://test.com/schema.json"
    }

    fetch_json_schema(schema_reference, "source_file.json")
    mock_fetch_json_schema_from_file.assert_called_once_with(
        schema_reference, "source_file.json"
    )
