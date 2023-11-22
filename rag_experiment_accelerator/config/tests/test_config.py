import pytest
from unittest.mock import patch
import json
import openai

from rag_experiment_accelerator.config.config import Config
from rag_experiment_accelerator.config.auth import OpenAICredentials


class TestEmbeddingModel:
    def __init__(self, model_name, dimension):
        self.model_name = model_name
        self.dimension = dimension

    def try_retrieve_model(self):
        pass

    def get_embeddings(self, text):
        return [1, 2, 3]



@patch("rag_experiment_accelerator.config.config.Config._try_retrieve_model")
@patch("rag_experiment_accelerator.config.config.OpenAICredentials.from_env")
@patch("rag_experiment_accelerator.config.config.EmbeddingModelFactory.create")
def test_config_init(
    mock_embeddings_model_factory,
    mock_from_env,
    mock_openai_model_retrieve,
):
    # Load mock config data from a json file
    with open(
        "rag_experiment_accelerator/config/tests/data/test_config.json", "r"
    ) as file:
        mock_config_data = json.load(file)
    mock_from_env.return_value = OpenAICredentials(
        openai_api_type="azure",
        openai_api_key="somekey",
        openai_api_version="v1",
        openai_endpoint="http://example.com",
    )
    mock_embeddings_model_factory.side_effect = [TestEmbeddingModel("", 123), TestEmbeddingModel("", 123) ]

    mock_openai_model_retrieve.return_value = {
        "status": "succeeded",
        "capabilities": {
            "embeddings": True,
            "inference": True,
            "chat_completion": True,
        },
    }
    config = Config("rag_experiment_accelerator/config/tests/data/test_config.json")
    assert config.NAME_PREFIX == mock_config_data["name_prefix"]
    assert config.CHUNK_SIZES == mock_config_data["chunking"]["chunk_size"]
    assert config.OVERLAP_SIZES == mock_config_data["chunking"]["overlap_size"]
    assert config.EF_CONSTRUCTIONS == mock_config_data["ef_construction"]
    assert config.EF_SEARCHES == mock_config_data["ef_search"]
    assert config.RERANK == mock_config_data["rerank"]
    assert config.RERANK_TYPE == mock_config_data["rerank_type"]
    assert config.LLM_RERANK_THRESHOLD == mock_config_data["llm_re_rank_threshold"]
    assert config.CROSSENCODER_AT_K == mock_config_data["cross_encoder_at_k"]
    assert config.CROSSENCODER_MODEL == mock_config_data["crossencoder_model"]
    assert config.SEARCH_VARIANTS == mock_config_data["search_types"]
    assert config.METRIC_TYPES == mock_config_data["metric_types"]
    assert config.CHAT_MODEL_NAME == mock_config_data["chat_model_name"]
    assert config.TEMPERATURE == mock_config_data["openai_temperature"]
    assert (
        config.SEARCH_RELEVANCY_THRESHOLD
        == mock_config_data["search_relevancy_threshold"]
    )
    assert config.DATA_FORMATS == mock_config_data["data_formats"]
    assert (
        config.EVAL_DATA_JSONL_FILE_PATH
        == mock_config_data["eval_data_jsonl_file_path"]
    )
    assert (
        mock_openai_model_retrieve.called
    )  # Ensure that the OpenAI model is retrieved


@pytest.mark.parametrize(
    "model_status, capabilities, tags, raises_exception",
    [
        (
            "succeeded",
            {"chat_completion": True, "inference": True},
            ["chat_completion", "inference"],
            False,
        ),
        (
            "failed",
            {"chat_completion": True, "inference": True},
            ["chat_completion", "inference"],
            True,
        ),
        (
            "succeeded",
            {"chat_completion": False, "inference": True},
            ["chat_completion", "inference"],
            True,
        ),
        (
            None,
            None,
            ["chat_completion", "inference"],
            True,
        ),
    ],
)
def test_try_retrieve_model(model_status, capabilities, tags, raises_exception):
    if model_status is not None:
        with patch(
            "rag_experiment_accelerator.config.config.openai.Model.retrieve"
        ) as mock_retrieve:
            mock_model = {
                "status": model_status,
                "capabilities": {
                    "chat_completion": capabilities["chat_completion"],
                    "inference": capabilities["inference"],
                    "embeddings": True,
                    "test": True,
                },
            }
            mock_retrieve.return_value = mock_model

            if raises_exception:
                with pytest.raises(ValueError):
                    config = Config("rag_experiment_accelerator/config/tests/data/test_config.json")
                    config.OpenAICredentials.OPENAI_API_TYPE = "azure"
                    config._try_retrieve_model("model_name", tags)
            else:
                config = Config("rag_experiment_accelerator/config/tests/data/test_config.json")
                config.OpenAICredentials.OPENAI_API_TYPE = "azure"
                result = config._try_retrieve_model("model_name", tags)
                assert result == mock_model
    else:
        with patch(
            "rag_experiment_accelerator.config.config.openai.Model.retrieve",
            side_effect=openai.error.InvalidRequestError("Test error", "404"),
        ):
            with pytest.raises(ValueError):
                config = Config("rag_experiment_accelerator/config/tests/data/test_config.json")
                config._try_retrieve_model("model_name", tags)


@pytest.mark.parametrize(
    "api_type, chat_model_name, embedding_model_name, chat_tags, embedding_tags",
    [
        ("openai", "gpt-3", None, ["chat_completion", "inference"], None),
        ("azure", None, "bert", None, ["embeddings", "inference"]),
        (
            "openai",
            "gpt-3",
            "bert",
            ["chat_completion", "inference"],
            ["embeddings", "inference"],
        ),
    ],
)
def test_check_deployment(
    api_type,
    chat_model_name,
    embedding_model_name,
    chat_tags,
    embedding_tags,
):
    with (
        patch("rag_experiment_accelerator.config.config.Config._try_retrieve_model") as mock_try_retrieve_model,
        patch("rag_experiment_accelerator.config.config.EmbeddingModelFactory.create") as mock_embeddings_model_factory,
    ):
        mock_try_retrieve_model.return_value = None  # Adjust as needed
        embedding_models = [TestEmbeddingModel("", 123), TestEmbeddingModel("", 123) ]
        mock_embeddings_model_factory.side_effect = embedding_models

        config = Config("rag_experiment_accelerator/config/tests/data/test_config.json")
        config.OpenAICredentials.OPENAI_API_TYPE = api_type
        config.CHAT_MODEL_NAME = chat_model_name

        config._check_deployment()
        if chat_model_name:
            mock_try_retrieve_model.assert_called_with(chat_model_name, tags=chat_tags)

        # TODO: make sure embedding models get called 
