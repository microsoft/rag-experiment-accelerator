from unittest.mock import patch
from rag_experiment_accelerator.utils.auth import OpenAICredentials


@patch("rag_experiment_accelerator.utils.auth.get_env_var")
def test_from_env_openai_credentials(mock_get_env_var):
    mock_get_env_var.side_effect = ["azure", "envkey", "v1", "http://envexample.com"]

    creds = OpenAICredentials.from_env()

    assert creds.OPENAI_API_TYPE == "azure"
    assert creds.OPENAI_API_KEY == "envkey"
    assert creds.OPENAI_API_VERSION == "v1"
    assert creds.OPENAI_ENDPOINT == "http://envexample.com"
