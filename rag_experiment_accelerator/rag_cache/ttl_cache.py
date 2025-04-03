import time
from cachetools import TTLCache
from collections import defaultdict
from queue import Queue

from rag_experiment_accelerator.rag_cache.cache_entry import CacheEntry


class CallbackTTLCache(TTLCache):
    def __init__(self, maxsize, ttl, eviction_callback=None, **kwargs):
        super().__init__(maxsize=maxsize, ttl=ttl, **kwargs)
        self.eviction_callback = eviction_callback

    def popitem(self):
        """
        Override popitem to call eviction_callback when an item is evicted.
        This method is called when an item is removed from the cache, either due to TTL or LRU.
        """
        key, value = super().popitem()  # Get the evicted key and value

        # Call the eviction callback, if provided
        if self.eviction_callback:
            self.eviction_callback([key])  # Pass the key and value to the callback

        return key, value