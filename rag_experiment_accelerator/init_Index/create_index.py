from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    CharFilter,
    CorsOptions,
    HnswParameters,
    HnswVectorSearchAlgorithmConfiguration,
    LexicalTokenizer,
    PrioritizedFields,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    SimpleField,
    TokenFilter,
    VectorSearch,
    VectorSearchProfile,
)

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def create_acs_index(
    service_endpoint,
    index_name,
    key,
    dimension,
    ef_construction,
    ef_search,
    analyzers,
):
    try:

        credential = AzureKeyCredential(key)

        # Apply checks on analyzer settings. Search analyzer and index analyzer must be set together
        index_analyzer = (
            analyzers["index_analyzer_name"]
            if analyzers.get("index_analyzer_name")
            else ""
        )
        search_analyzer = (
            analyzers["search_analyzer_name"]
            if analyzers.get("search_analyzer_name")
            else ""
        )
        # Analyzer can only be used if neither search analyzer or index analyzer are set
        if analyzers.get("analyzer_name") and (analyzers.get("search_analyzer_name") or analyzers.get("index_analyzer_name")):
            raise ValueError(
                "analyzer_name should be empty if either search_analyzer_name or index_analyzer_name is not empty")
        analyzer = analyzers.get("analyzer_name") or ""

        # Create a search index
        index_client = SearchIndexClient(
            endpoint=service_endpoint, credential=credential
        )
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(
                name="content",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
            SearchableField(
                name="title",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
            SearchableField(
                name="summary",
                type=SearchFieldDataType.String,
                searchable=True,
                retrievable=True,
            ),
            SearchableField(
                name="filename",
                type=SearchFieldDataType.String,
                filterable=True,
                searchable=False,
                retrievable=True,
            ),
            SearchableField(
                name="description",
                type=SearchFieldDataType.String,
                index_analyzer_name=index_analyzer,
                search_analyzer_name=search_analyzer,
            ),
            SearchableField(
                name="text",
                type=SearchFieldDataType.String,
                searchable=True,
                analyzer_name=search_analyzer,
            ),
            SearchField(
                name="contentVector",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=int(dimension),
                vector_search_profile="my-vector-search-profile",
            ),
            SearchField(
                name="contentTitle",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=int(dimension),
                vector_search_profile="my-vector-search-profile",
            ),
            SearchField(
                name="contentSummary",
                type=SearchFieldDataType.Collection(
                    SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=int(dimension),
                vector_search_profile="my-vector-search-profile",
            ),
            SearchField(
                name="contentDescription",
                type=SearchFieldDataType.String,
                sortable=True,
                filterable=True,
                facetable=True,
                analyzer_name=analyzer,
            ),
        ]

        vector_search = VectorSearch(
            algorithms=[
                HnswVectorSearchAlgorithmConfiguration(
                    name="my-vector-config",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=int(ef_construction),
                        ef_search=int(ef_search),
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

        semantic_config = SemanticConfiguration(
            name="my-semantic-config",
            prioritized_fields=PrioritizedFields(
                prioritized_content_fields=[
                    SemanticField(field_name="content")]
            ),
        )

        # Create the semantic settings with the configuration
        semantic_settings = SemanticSettings(configurations=[semantic_config])

        # Define a custom tokenizer, token filter and char filter
        tokenizers = []
        token_filters = []
        char_filters = []
        if analyzers.get("tokenizers"):
            tokenizers = [
                LexicalTokenizer(
                    name=tokenizer["name"],
                    token_chars=tokenizer["token_chars"],
                )
                for tokenizer in analyzers["tokenizers"]
            ]
        if analyzers.get("token_filters"):
            # token_filters = [LexicalTokenFilter(name=analyzers["token_filters"]["name"], odatatype="#Microsoft.Azure.Search.AsciiFoldingTokenFilter")]
            token_filters = [
                TokenFilter(name="lowercase"),
                TokenFilter(name="asciifolding"),
            ]
        if analyzers.get("char_filters"):
            char_filters = [
                CharFilter(
                    name=char_filter["name"],
                    odatatype="#Microsoft.Azure.Search.MappingCharFilter",
                    mappings=char_filter["mappings"],
                )
                for char_filter in analyzers["char_filters"]
            ]

        cors_options = CorsOptions(
            allowed_origins=["*"], max_age_in_seconds=60)
        scoring_profiles = []

        # Create the search index with the semantic, tokenizer, and filter settings
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
            semantic_settings=semantic_settings,
            scoring_profiles=scoring_profiles,
            cors_options=cors_options,
            tokenizers=tokenizers,
            token_filters=token_filters,
            char_filters=char_filters,
        )
        result = index_client.create_or_update_index(index)
        logger.info(f"{result.name} created")

    except Exception as e:
        raise ValueError("An error occurred while creating index: " + str(e))


def delete_index(index_client, index_name):
    try:
        index_client.delete_index(index_name)
    except Exception as e:
        print(f"An error occurred while deleting index: {e}")
