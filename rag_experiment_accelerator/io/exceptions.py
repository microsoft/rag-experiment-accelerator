class WriteException(Exception):
    def __init__(self, path: str, e: Exception):
        super().__init__(
            f"Unable to write to file to path: {path}. Please ensure"
            " you have the proper permissions to write to the file.",
            e,
        )


class CopyException(Exception):
    def __init__(self, src: str, dest: str, e: Exception):
        super().__init__(
            f"Unable to copy file from {src} to {dest}. Please ensure"
            " you have the proper permissions to copy the file.",
            e,
        )
