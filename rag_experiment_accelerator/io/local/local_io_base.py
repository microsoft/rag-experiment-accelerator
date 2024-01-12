import os


class LocalIOBase:
    def exists(self, path: str) -> bool:
        if os.path.exists(path):
            return True
        return False
