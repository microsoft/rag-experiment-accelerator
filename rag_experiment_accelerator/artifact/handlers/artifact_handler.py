import time

from rag_experiment_accelerator.loaders.exceptions import UnsupportedFileFormatException
from rag_experiment_accelerator.loaders.typing import U
from rag_experiment_accelerator.writers.typing import V
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class ArtifactHandler:
    def __init__(self, directory: str, writer: V, loader: U) -> None:
        self.directory = directory
        self.archive_dir = f"{self.directory}/archive"
        self._writer = writer
        self._directory_structure_initialized = False
        self._artifact_dir_initialized = False
        self._loader = loader

    def load(self, filename: str, **kwargs) -> list:
        path = f"{self.directory}/{filename}"
        # check loader can handle the file
        if not self._loader.can_handle(path):
            raise UnsupportedFileFormatException(path)

        # ensure file exists
        if not self._loader.exists(path):
            logger.error(f"Unable to load artifacts. Artifact file not found: {path}")
            return None

        # load artifacts
        return self._loader.load(path=path, **kwargs)

    def archive(self, filename: str) -> str | None:
        src = f"{self.directory}/{filename}"
        # archive if file exists, else no-op
        if self._writer.exists(src):
            # timestamp filename in archive dir
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            dest = f"{self.archive_dir}/{timestamp}-{filename}"

            # copy file to archive
            self._writer.copy(src, dest)

            # delete original file
            self._writer.delete(src)
            logger.info(f"Archived previous artifacts to {dest}")

            return dest
