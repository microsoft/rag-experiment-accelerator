from unittest.mock import patch, MagicMock
import json
import urllib
from rag_experiment_accelerator.embedding.custom_embedding_model import (
    CustomEmbeddingModel,
)
import ssl


def test_can_set_embedding_dimension():
    environment = MagicMock()
    model = CustomEmbeddingModel("custom-embedding-model", environment, 123)
    assert model.dimension == 123


def test_prepare_request_success():
    environment = MagicMock()
    environment.azure_model_api_key = "api_key"
    model = CustomEmbeddingModel("custom-embedding-deployment", environment)

    body = {"text": "Hello world"}
    headers, prepared_body = model.prepare_request(body)

    expected_headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer api_key",
        "azureml-model-deployment": "custom-embedding-deployment",
    }
    expected_body = str.encode(json.dumps({"input": body}))

    assert headers == expected_headers
    assert prepared_body == expected_body


@patch("urllib.request.urlopen")
def test_make_request_success(mock_urlopen):
    environment = MagicMock()
    environment.azure_model_api_endpoint = "http://fake-endpoint"
    model = CustomEmbeddingModel("custom-embedding-model", environment)

    mock_response = MagicMock()
    mock_response.read.return_value = json.dumps([0.1, 0.2, 0.3]).encode("utf-8")
    mock_urlopen.return_value = mock_response

    headers = {"Content-Type": "application/json"}
    body = b'{"input": {"text": "Hello world"}}'

    result = model.make_request(body, headers)
    assert result == [0.1, 0.2, 0.3]


@patch("urllib.request.urlopen")
def test_make_request_http_error(mock_urlopen):
    environment = MagicMock()
    environment.azure_model_api_endpoint = "http://fake-endpoint"
    model = CustomEmbeddingModel("custom-embedding-model", environment)

    mock_urlopen.side_effect = urllib.error.HTTPError(
        url=None, code=500, msg="Internal Server Error", hdrs=None, fp=None
    )

    headers = {"Content-Type": "application/json"}
    body = b'{"input": {"text": "Hello world"}}'

    try:
        model.make_request(body, headers)
    except urllib.error.HTTPError as e:
        assert e.code == 500


@patch(
    "rag_experiment_accelerator.embedding.custom_embedding_model.CustomEmbeddingModel.make_request"
)
def test_generate_embedding_success(mock_make_request):
    environment = MagicMock()
    model = CustomEmbeddingModel("custom-embedding-model", environment)

    mock_make_request.return_value = [0.1, 0.2, 0.3]

    result = model.generate_embedding("Hello world")
    assert result == [0.1, 0.2, 0.3]


def test_allow_self_signed_http_true():
    environment = MagicMock()
    model = CustomEmbeddingModel("custom-embedding-model", environment)

    model.allowSelfSignedHttps(True)
    assert ssl._create_default_https_context == ssl._create_unverified_context


def test_allow_self_signed_http_false():
    environment = MagicMock()
    model = CustomEmbeddingModel("custom-embedding-model", environment)

    model.allowSelfSignedHttps(False)
    assert ssl._create_default_https_context == ssl.create_default_context
