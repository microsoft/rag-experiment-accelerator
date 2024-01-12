from typing import TypeVar

from rag_experiment_accelerator.io.loader import Loader
from rag_experiment_accelerator.io.writer import Writer


V = TypeVar("V", bound=Writer)
U = TypeVar("U", bound=Loader)
