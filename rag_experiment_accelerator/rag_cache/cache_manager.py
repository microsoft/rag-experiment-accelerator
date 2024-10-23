import threading
import time
from cachetools import TTLCache
from collections import OrderedDict, defaultdict
from queue import Queue

from rag_experiment_accelerator.rag_cache.cache_entry import *
from rag_experiment_accelerator.rag_cache.ttl_cache import CallbackTTLCache


class LruLfuTTLCache:
    def __init__(self, capacity, ttl=1 * 60, eviction_callback=None):
        self.capacity = capacity
        self.ttl = ttl  # 1 hours by default (in seconds)

        self.eviction_callback = eviction_callback

        # TTLCache to automatically evict items after `ttl` seconds (LRU + TTL)
        self.lru_cache = CallbackTTLCache(maxsize=capacity, ttl=ttl, eviction_callback=eviction_callback)

        # LFU cache to track frequency of use
        self.lfu_cache = defaultdict(int)

    def add_or_update_cache_batch(self, doc_ids):
        """
        Adds or updates a batch of documents in the cache.

        Args:
            doc_ids: A list of document IDs.
        """

        for doc_id in doc_ids:
            self.add_or_update_cache(doc_id)

    def add_or_update_cache(self, doc_id):
        if doc_id in self.lru_cache:
            # Update access time and frequency for LRU + LFU
            print(f"Accessing doc_id: {doc_id}")
            cache_entry = self.lru_cache[doc_id]
            cache_entry.access()
            self.lfu_cache[doc_id] += 1
        else:
            # Add new entry if it doesn't exist in cache
            if len(self.lru_cache) >= self.capacity:
                self.evict()  # Evict least used entry if cache is full

            entry = CacheEntry(doc_id, self.ttl)
            self.lru_cache[doc_id] = entry
            self.lfu_cache[doc_id] = 1

    def evict(self):
        """
        Evict the least frequently used and least recently used items if necessary.
        Combines LRU, LFU, and TTL for eviction policy.
        """
        lfu_candidate = min(self.lfu_cache, key=self.lfu_cache.get)

        if lfu_candidate:
            # Remove from both LRU and LFU
            print(f"Evicting id: {lfu_candidate} due to being least frequently used.")
            cache_evicted_doc = self.lru_cache.pop(lfu_candidate, None)
            self.lfu_cache.pop(lfu_candidate, None)

            if cache_evicted_doc:
                self.removal_queue.put([cache_evicted_doc])

    def contains_key(self, doc_id):
        """
        Check if a document is in the cache.
        :param doc_id: Document ID to check.
        :return: True if the document is in the cache, False otherwise.
        """
        return doc_id in self.lru_cache

    def get_key(self, doc_id):
        """
        Access a document in the cache (LRU+LFU).
        :param doc_id: The document ID to access.
        :return: The CacheEntry associated with the document.
        """
        if doc_id in self.lru_cache:
            self.lfu_cache[doc_id] += 1  # Update LFU count
            return self.lru_cache[doc_id]
        return None
