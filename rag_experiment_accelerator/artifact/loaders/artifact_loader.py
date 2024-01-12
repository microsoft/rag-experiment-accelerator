from typing import Generic

from rag_experiment_accelerator.artifact.models.typing import T
from rag_experiment_accelerator.loaders.exceptions import (
    UnsupportedFileFormatException,
)
from rag_experiment_accelerator.loaders.typing import U
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class ArtifactLoader(Generic[T]):
    def __init__(self, class_to_load: type[T], directory: str, loader: U) -> None:
        self.directory = directory
        self.archive_dir = f"{self.directory}/archive"
        self._class_to_load: type[T] = class_to_load
        self.loader = loader

    def load_artifacts(self, filename: str, **kwargs) -> list[T] | None:
        path = f"{self.directory}/{filename}"
        # check loader can handle the file
        if not self.loader.can_handle(path):
            raise UnsupportedFileFormatException(path)

        # ensure file exists
        if not self.loader.exists(path):
            logger.error(f"Unable to load artifacts. Artifact file not found: {path}")
            return None

        # load artifacts
        content = self.loader.load(path=path, **kwargs)
        data_load: list[T] = [self._class_to_load.create(d) for d in content]
        return data_load
