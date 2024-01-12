import time
from typing import Generic
from rag_experiment_accelerator.artifact.models.typing import T
from rag_experiment_accelerator.writers.typing import V
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


class ArtifactWriter(Generic[T]):
    """
    A class that provides functionality to write and archive artifacts.

    Args:
        directory (str): The directory where the artifacts will be saved.
        writer (V): The writer object used to perform the writing operations.

    Attributes:
        directory (str): The directory where the artifacts will be saved.
        archive_dir (str): The directory where the archived artifacts will be stored.
    """

    def __init__(self, directory: str, writer: V) -> None:
        self.directory = directory
        self.archive_dir = f"{self.directory}/archive"
        self._writer = writer
        self._directory_structure_initialized = False
        self._artifact_dir_initialized = False

    def archive_artifact(self, filename: str) -> str | None:
        """
        Archives the specified artifact by copying it to the archive directory.

        Args:
            filename (str): The name of the artifact file to be archived.

        Returns:
            str | None: The path of the archived artifact if successful, None otherwise.
        """
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

    def save_artifact(self, data: T, filename: str, **kwargs) -> None:
        """
        Saves the specified artifact by writing it to the specified file.

        Args:
            data (T): The data to be saved as the artifact.
            filename (str): The name of the file to save the artifact.
            **kwargs: Additional keyword arguments to be passed to the writer's write method.
        """
        path = f"{self.directory}/{filename}"
        self._writer.write(path, data.to_dict(), **kwargs)
