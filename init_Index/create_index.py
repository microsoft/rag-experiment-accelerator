import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    SearchIndex,
    SemanticConfiguration,
    PrioritizedFields,
    SemanticField,
    SearchField,
    SemanticSettings,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
)

from utils.logging import get_logger
logger = get_logger(__name__)


def create_acs_index(service_endpoint,
                     index_name,
                     key,
                     dimension,
                     efconstruction,
                     efsearch):
    """
    Creates a search index in Azure Cognitive Search with the specified parameters.

    Args:
        service_endpoint (str): The endpoint URL for the Azure Cognitive Search service.
        index_name (str): The name of the search index to create.
        key (str): The API key for the Azure Cognitive Search service.
        dimension (int): The number of dimensions to use for vector search.
        efconstruction (int): The maximum number of nodes to be visited during index construction.
        efsearch (int): The maximum number of nodes to be visited during a search query.

    Returns:
        None
    """

    credential = AzureKeyCredential(key)

    # Create a search index
    index_client = SearchIndexClient(endpoint=service_endpoint, credential=credential)
    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String,
                        searchable=True, retrievable=True),
        SearchableField(name="title", type=SearchFieldDataType.String,
                        searchable=True, retrievable=True),
        SearchableField(name="summary", type=SearchFieldDataType.String,
                        searchable=True, retrievable=True),
        SearchableField(name="filename", type=SearchFieldDataType.String,
                        filterable=True, searchable=False, retrievable=True),
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=int(dimension),
                    vector_search_configuration="my-vector-config"),
        SearchField(name="contentTitle", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=int(dimension),
                    vector_search_configuration="my-vector-config"),
        SearchField(name="contentSummary", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=int(dimension),
                    vector_search_configuration="my-vector-config"),

    ]

    vector_search = VectorSearch(
        algorithm_configurations=[
            HnswVectorSearchAlgorithmConfiguration(
                name="my-vector-config",
                kind="hnsw",
                hnsw_parameters={
                    "m": 4,
                    "efConstruction": int(efconstruction),
                    "efSearch": int(efsearch),
                    "metric": "cosine"
                }
            )
        ]
    )

    semantic_config = SemanticConfiguration(
        name="my-semantic-config",
        prioritized_fields=PrioritizedFields(
            prioritized_content_fields=[SemanticField(field_name="content")]
        )
    )

    # Create the semantic settings with the configuration
    semantic_settings = SemanticSettings(configurations=[semantic_config])

    # Create the search index with the semantic settings
    index = SearchIndex(name=index_name, fields=fields,
                        vector_search=vector_search, semantic_settings=semantic_settings)
    result = index_client.create_or_update_index(index)
    logger.info(f' {result.name} created')
