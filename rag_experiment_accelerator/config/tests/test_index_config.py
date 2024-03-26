from unittest.mock import MagicMock

from rag_experiment_accelerator.config.index_config import IndexConfig


def test_index_config_to_index_name():
    mock_embedding_model = MagicMock()
    mock_embedding_model.name = "modelname"

    index_config = IndexConfig(
        index_name_prefix="prefix",
        chunk_size=1,
        overlap=2,
        embedding_model=mock_embedding_model,
        ef_construction=3,
        ef_search=4,
        generate_summary=False,
        generate_title=False,
        override_content_with_summary=False,
    )

    assert index_config.index_name() == "prefix_1_2_modelname_3_4"


def test_index_name_to_index_config():
    index_name = "prefix_1_2_modelname_3_4"
    mock_embedding_model = MagicMock()
    mock_embedding_model.name = "modelname"
    config = MagicMock()
    config._find_embedding_model_by_name.return_value = mock_embedding_model

    index_config = IndexConfig.from_index_name(index_name, config)

    assert index_config.index_name_prefix == "prefix"
    assert index_config.chunk_size == 1
    assert index_config.overlap == 2
    assert index_config.embedding_model.name == "modelname"
    assert index_config.ef_construction == 3
    assert index_config.ef_search == 4
