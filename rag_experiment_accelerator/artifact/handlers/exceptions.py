class LoadException(Exception):
    def __init__(self, path: str):
        super().__init__(
            f"Cannot load at path: {path}. Please ensure it is supported by the loader."
        )
