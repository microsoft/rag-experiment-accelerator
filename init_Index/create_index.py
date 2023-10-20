

from azure.core.credentials import AzureKeyCredential  
from azure.search.documents.indexes import SearchIndexClient  
from azure.search.documents.indexes.models import (
    CharFilter,  
    ComplexField, 
    CorsOptions, 
    SearchIndex, 
    ScoringProfile, 
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
    LexicalTokenizer,
  #  LexicalTokenFilter,  
    SemanticSettings,  
    TokenFilter,
    VectorSearch,  
    HnswVectorSearchAlgorithmConfiguration,  
)
 

def create_acs_index(service_endpoint,
                    index_name,
                    key,
                    dimension,
                    efconstruction,
                    efsearch,
                    analyzers   ):

    credential = AzureKeyCredential(key)
    analyzer = analyzers["analyzer_name"]
    index_analyzer = analyzers["index_analyzer_name"]
    search_analyzer = analyzers["search_analyzer_name"]
    # Create a search index
    index_client = SearchIndexClient(
        endpoint=service_endpoint, credential=credential)
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
    #    SearchableField(name="description", type=SearchFieldDataType.String, 
    #                    index_analyzer_name=index_analyzer),                        
    #    SearchableField(name='text', type=SearchFieldDataType.String, 
    #                    searchable=True, analyzer_name=search_analyzer),                
        SearchField(name="contentVector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=int(dimension), vector_search_configuration="my-vector-config"),
        SearchField(name="contentTitle", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=int(dimension), vector_search_configuration="my-vector-config"),
        SearchField(name="contentSummary", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=int(dimension), vector_search_configuration="my-vector-config"),
        SearchField(name="contentDescription", type=SearchFieldDataType.String, key=True,
                    sortable=True, filterable=True, facetable=True, analyzer_name=analyzer),                    

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

    # Define a custom tokenizer, token filter and char filter
    tokenizers = []
    token_filters = []
    char_filters = []
    if analyzers["tokenizers"]:
        tokenizers = [LexicalTokenizer(name=analyzers["tokenizers"]["name"], token_chars=["letter", "digit"])]
    if analyzers["token_filters"]:
       # token_filters = [LexicalTokenFilter(name=analyzers["token_filters"]["name"], odatatype="#Microsoft.Azure.Search.AsciiFoldingTokenFilter")]
        token_filters = [TokenFilter(name="lowercase"),TokenFilter(name="asciifolding")]
    if analyzers["char_filters"]:
        char_filters = [CharFilter(name=analyzers["char_filters"]["name"], odatatype="#Microsoft.Azure.Search.MappingCharFilter", mappings=[])]

    cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=60)
    scoring_profiles = []

    # Create the search index with the semantic settings
    index = SearchIndex(name=index_name, fields=fields,
                        vector_search=vector_search,
                        semantic_settings=semantic_settings,
                        scoring_profiles=scoring_profiles,
                        cors_options=cors_options,
                        tokenizers=tokenizers,
                        token_filters=token_filters,
                        char_filters=char_filters)
    result = index_client.create_or_update_index(index)
    print(f' {result.name} created')