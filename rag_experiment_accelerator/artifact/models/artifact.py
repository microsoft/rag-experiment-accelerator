from abc import ABC
from typing import Self


class Artifact(ABC):
    def to_dict(self):
        return self.__dict__

    @classmethod
    def create(cls, data: dict | str) -> Self:
        if isinstance(data, dict):
            return cls(**data)
        else:
            return cls(data)
