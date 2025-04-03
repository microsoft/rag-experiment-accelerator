import time

class CacheEntry:
    def __init__(self, doc_id, ttl):
        self.doc_id = doc_id
        self.last_access_time = time.time() # For LRU (last access time)
        self.ttl = time.time() + ttl  # Expiration time

    def access(self):
        self.last_access_time = time.time()

    def is_expired(self):
        return time.time() > self.ttl
    
    def get_doc_id(self):
        return self.doc_id