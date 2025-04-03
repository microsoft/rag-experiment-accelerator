from azure.search.documents.indexes.models import (
    CorsOptions,
    ExhaustiveKnnParameters,
    ExhaustiveKnnVectorSearchAlgorithmConfiguration,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def create_acs_cache_index(
        index_client,
        dimension,
        index_name
):
    try:
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="prompt_text",
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=True,
                retrievable=True,
            ),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=True,
                retrievable=True,
            ),
            SearchField(
                name="prompt_embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                retrievable=True,
                vector_search_dimensions=int(dimension),
                vector_search_profile="my-vector-search-profile",
            ),
            SearchableField(
                name="knowledge_base_docids",
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=True,
                retrievable=True,
            )
        ]

        vector_search = VectorSearch(
            algorithms=[
                ExhaustiveKnnVectorSearchAlgorithmConfiguration(
                    name="my-vector-config",
                    parameters=ExhaustiveKnnParameters(
                        metric="cosine",
                    ),
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="my-vector-search-profile", algorithm="my-vector-config"
                )
            ],
        )

        cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
        scoring_profiles = []

        # Create the search index with the semantic, tokenizer, and filter settings

        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            scoring_profiles=scoring_profiles,
            cors_options=cors_options,
        )
        index_client.delete_index(index)
        result = index_client.create_or_update_index(index)
        if result:
            logger.info("Index created or updated successfully.")
        else:
            logger.error("Failed to create or update index.")

    except Exception as e:
        raise ValueError(f"An error occurred while creating index [{index_name}]: {e}")


def get_rag_index_doc(doc_id, prompt_text, prompt_embedding, llm_text_content, knowledge_base_docids):
    document = {
        "id": doc_id,
        "prompt_text": prompt_text,
        "prompt_embedding": prompt_embedding,
        "content": llm_text_content,
        "knowledge_base_docids": knowledge_base_docids
    }
    logger.info("document is added ", document)
    return [document]

