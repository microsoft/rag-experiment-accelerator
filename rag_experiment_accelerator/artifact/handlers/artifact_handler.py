import time

from rag_experiment_accelerator.artifact.handlers.exceptions import LoaderException
from rag_experiment_accelerator.io.typing import U, V
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class ArtifactHandler:
    def __init__(self, data_location: str, writer: V, loader: U) -> None:
        self.data_location = data_location
        self.archive_location = f"{self.data_location}/archive"
        self._writer = writer
        self._loader = loader

    def load(self, name: str, **kwargs) -> list:
        path = f"{self.data_location}/{name}"
        # check loader can handle the file
        if not self._loader.can_handle(path):
            raise LoaderException(
                f"Cannot load file at path: {path}. Please ensure the file is supported by the loader."
            )

        # load artifacts
        logger.info(f"Loading artifacts from path: {path}")
        loaded_data = self._loader.load(path=path, **kwargs)
        if len(loaded_data) == 0:
            raise LoaderException(
                f"No data loaded from path: {path}. Please ensure the file is not empty."
            )
        return loaded_data

    def handle_archive(self, name: str) -> str | None:
        src = f"{self.data_location}/{name}"
        logger.debug(f"Attemping to archive file at path: {src}")
        # archive if file exists, else no-op
        if self._writer.exists(src):
            # timestamp filename in archive dir
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            dest = f"{self.archive_location}/{timestamp}-{name}"

            # copy file to archive
            self._writer.copy(src, dest)

            # delete original file
            self._writer.delete(src)
            logger.info(f"Archived previous artifacts to {dest}")

            return dest
        logger.debug(f"No file to archive at path: {src}")

    def save_dict(self, data: dict, name: str, **kwargs):
        path = f"{self.data_location}/{name}"
        logger.info(f"Saving artifacts to path: {path}")
        self._writer.write(path, data, **kwargs)
