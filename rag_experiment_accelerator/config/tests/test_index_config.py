from rag_experiment_accelerator.config.chunking_config import ChunkingConfig
from rag_experiment_accelerator.config.embedding_model_config import (
    EmbeddingModelConfig,
)
from rag_experiment_accelerator.config.index_config import IndexConfig


def test_index_config_to_index_name():
    index_config = IndexConfig(
        index_name_prefix="prefix",
        ef_construction=3,
        ef_search=4,
        chunking=ChunkingConfig(
            preprocess=False,
            chunk_size=1,
            overlap_size=2,
            generate_summary=False,
            generate_title=False,
            override_content_with_summary=False,
        ),
        embedding_model=EmbeddingModelConfig(type="type", model_name="modelname"),
    )

    assert (
        index_config.index_name()
        == "idx-prefix_efc-3_efs-4_emn-modelname_p-0_cs-1_o-2_t-0_s-0_oc-0"  # todo: update with indexname
    )


def test_index_name_to_index_config():
    index_name = "idx-prefix_efc-3_efs-4_emn-modelname_p-0_cs-1_o-2_t-0_s-0_oc-0"  # todo: update with indexname

    index_config = IndexConfig.from_index_name(index_name)

    assert index_config.index_name_prefix == "prefix"
    assert index_config.chunking.chunk_size == 1
    assert index_config.chunking.overlap_size == 2
    assert index_config.embedding_model.model_name == "modelname"
    assert index_config.ef_construction == 3
    assert index_config.ef_search == 4
