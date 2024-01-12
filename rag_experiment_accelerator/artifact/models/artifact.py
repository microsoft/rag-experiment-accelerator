from abc import ABC
from typing import Self


class Artifact(ABC):
    """
    Base class for artifacts.
    """

    def to_dict(self):
        """
        Convert the artifact to a dictionary representation.

        Returns:
            dict: The dictionary representation of the artifact.
        """
        return self.__dict__

    @classmethod
    def create(cls, data: dict | str) -> Self:
        """
        Create an instance of the artifact.

        Args:
            data (dict | str): The data used to create the artifact. If a dictionary is provided,
                the artifact will be created using the dictionary's key-value pairs as arguments.
                If a string is provided, the artifact will be created using the string as a single argument.

        Returns:
            Self: An instance of the artifact.
        """
        if isinstance(data, dict):
            return cls(**data)
        else:
            return cls(data)
