import time
from typing import Generic
from rag_experiment_accelerator.artifact.models.typing import T
from rag_experiment_accelerator.writers.typing import V
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class ArtifactWriter(Generic[T]):
    def __init__(self, directory: str, writer: V) -> None:
        self.directory = directory
        self.archive_dir = f"{self.directory}/archive"
        self._writer = writer
        self._directory_structure_initialized = False
        self._artifact_dir_initialized = False

    def archive_artifact(self, filename: str) -> str | None:
        src = f"{self.directory}/{filename}"
        # archive if file exists, else no-op
        if self._writer.exists(src):
            # set up directory structure
            # if self._artifact_dir_initialized is False:
            #     self._writer.prepare_write(self.archive_dir)
            #     self._artifact_dir_initialized = True

            # timestamp filename in archive dir
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            dest = f"{self.archive_dir}/{timestamp}-{filename}"

            # copy file to archive
            self._writer.copy(src, dest)

            # delete original file
            self._writer.delete(src)
            logger.info(f"Archived previous artifacts to {dest}")

            return dest

    def save_artifact(self, data: T, filename: str, **kwargs) -> None:
        # set up directory structure
        # if self._directory_structure_initialized is False:
        #     self._writer.prepare_write(self.directory)
        #     self._directory_structure_initialized = True

        # write file
        path = f"{self.directory}/{filename}"
        self._writer.write(path, data.to_dict(), **kwargs)
