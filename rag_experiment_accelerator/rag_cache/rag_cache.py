import hashlib

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents._generated.models import RawVectorQuery
from azure.search.documents.indexes import SearchIndexClient
from rag_experiment_accelerator.rag_cache.rag_cache_index import create_acs_cache_index, get_rag_index_doc
from rag_experiment_accelerator.rag_cache.cache_util import *

from rag_experiment_accelerator.rag_cache.cache_manager import LruLfuTTLCache


class RagCache:
    _instances = {}  # Class variable to hold instances for different index names

    def __new__(cls, index_name, *args, **kwargs):
        if index_name not in cls._instances:
            cls._instances[index_name] = super(RagCache, cls).__new__(cls)
        return cls._instances[index_name]

    def __init__(self, environment, config, index_name, dim=768, max_doc=10000, threshold=0.75):
        # Ensure that the initialization runs only once per index
        if not hasattr(self, 'initialized'):
            """
            Initialize RagCache with dimension, service endpoint, index_config, maxDoc, and threshold.

            :param dim: Dimension of the cache (e.g., number of features or embeddings).
            :param max_doc: Maximum number of documents to cache.
            :param threshold: Cache threshold percentage (default is 75%).
            """
            self.environment = environment
            self.dim = dim
            self.max_doc = max_doc
            self.cur_doc_count = 0
            self.threshold = threshold
            self.config = config
            self.credential = AzureKeyCredential(environment.azure_search_admin_key)

            # Create a search index
            self.index_client = SearchIndexClient(
                endpoint=environment.azure_search_service_endpoint,
                credential=self.credential
            )

            self.index_name = index_name + "_rag_cache"

            self.search_client = SearchClient(
                endpoint=environment.azure_search_service_endpoint,
                index_name=self.index_name,
                credential=self.credential,
            )

            self.cache = LruLfuTTLCache(capacity=1000, ttl=60, eviction_callback=self.removeFromCache)
            self.initialized = True

    def create_rag_cache_index(self):
        # Call create_index during initialization
        create_acs_cache_index(self.index_client, self.dim, self.index_name)

    def addToCache(self, prompt_text, prompt_embedding, content, knowledgebase_docIds):
        doc_id = getHashDocId(prompt_text)
        print(" ==add doc id", doc_id)
        if len(prompt_embedding) != self.dim:
            raise ValueError(
                f"Embedding length {len(prompt_embedding)} does not match the expected dimension {self.dim}")

        if self.cur_doc_count < self.max_doc:
            doc = get_rag_index_doc(doc_id, prompt_text, prompt_embedding, content, knowledgebase_docIds)
            self.search_client.upload_documents(doc)
            print(f"Added to cache: {prompt_text}")
            self.cur_doc_count += 1
        else:
            print(f"Cache is full. Max documents allowed: {self.max_doc}")

    def getFromRagCache(self, query_embedding):
        if len(query_embedding) != self.dim:
            raise ValueError(
                f"Query embedding length {len(query_embedding)} does not match the expected dimension {self.dim}")
        formatted_search_results = None

        vector = RawVectorQuery(
            k=10,
            fields="prompt_embedding",
            vector=query_embedding,
            exhaustive=True
        )

        search_client = SearchClient(
            endpoint=self.environment.azure_search_service_endpoint,
            index_name=self.index_name,
            credential=self.credential,
        )

        try:
            results = search_client.search(
                search_text=None,
                vector_queries=[vector],
                select=["content", "prompt_text", "id"],
            )
            formatted_search_results = getResult(results, self.threshold)
        except Exception as e:
            print(str(e))
        return formatted_search_results

    def removeFromCache(self, doc_id):
        self.search_client.delete_documents([doc_id])

    # handling for updating the cache when cache source index docs are deleted
    def removeFromCache(self, knowledgebase_docIds):
        delete_doc = []
        try:
            for doc_id in knowledgebase_docIds:
                results = self.search_client.search(
                    search_text=knowledgebase_docIds,
                    select=["id"]
                )
                doc_ids = get_doc_id_from_result(results)
                if doc_ids:
                    delete_doc.extend(doc_ids)

        except Exception as e:
            print(str(e))

        self.removeFromCache(delete_doc)
