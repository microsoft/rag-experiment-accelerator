from rag_experiment_accelerator.config.chunking_config import ChunkingConfig
from rag_experiment_accelerator.config.embedding_model_config import (
    EmbeddingModelConfig,
)
from rag_experiment_accelerator.config.index_config import IndexConfig
from rag_experiment_accelerator.config.sampling_config import SamplingConfig


def test_index_config_to_index_name():
    index_config = IndexConfig(
        index_name_prefix="prefix",
        ef_construction=3,
        ef_search=4,
        chunking=ChunkingConfig(
            preprocess=False,
            chunk_size=1,
            chunking_strategy="abcd",
            overlap_size=2,
            generate_summary=False,
            generate_title=False,
            override_content_with_summary=False,
        ),
        embedding_model=EmbeddingModelConfig(
            type="type", model_name="modelname", dimension=100
        ),
        sampling=SamplingConfig(percentage=10),
    )

    assert (
        index_config.index_name()
        == "idx-prefix_efc-3_efs-4_em-modelname_sp-10_p-0_cs-1_st-abcd_o-2_t-0_s-0_oc-0_d-100"
    )


def test_index_name_to_index_config():
    index_name = "idx-prefix_efc-3_efs-4_em-modelname_sp-10_p-0_cs-1_st-abcd_o-2_t-0_s-0_oc-0_d-100"

    index_config = IndexConfig.from_index_name(index_name)

    assert index_config.index_name_prefix == "prefix"
    assert index_config.chunking.chunk_size == 1
    assert index_config.chunking.chunking_strategy == "abcd"
    assert index_config.chunking.overlap_size == 2
    assert index_config.embedding_model.model_name == "modelname"
    assert index_config.embedding_model.dimension == 100
    assert index_config.ef_construction == 3
    assert index_config.ef_search == 4


def test_index_name_to_index_config_shuffled_order():
    index_name = "idx-prefix_efc-3_efs-4_em-modelname_p-0_cs-1_st-abcd_o-2_t-0_s-0_oc-0_sp-10_d-100"

    index_config = IndexConfig.from_index_name(index_name)

    assert index_config.index_name_prefix == "prefix"
    assert index_config.chunking.chunk_size == 1
    assert index_config.chunking.chunking_strategy == "abcd"
    assert index_config.chunking.overlap_size == 2
    assert index_config.embedding_model.model_name == "modelname"
    assert index_config.embedding_model.dimension == 100
    assert index_config.ef_construction == 3
    assert index_config.ef_search == 4


def test_index_name_to_index_config_missing_property():
    index_name = (
        "idx-prefix_efc-3_efs-4_em-modelname_sp-10_p-0_st-basic_o-2_t-0_s-0_oc-0_d-100"
    )

    try:
        IndexConfig.from_index_name(index_name)
    except ValueError:
        assert True
    else:
        assert False, "Expected ValueError to be thrown"
