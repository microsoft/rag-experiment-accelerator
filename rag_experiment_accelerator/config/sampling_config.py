from dataclasses import dataclass
from rag_experiment_accelerator.config.base_config import BaseConfig


@dataclass
class SamplingConfig(BaseConfig):
    """A class to hold parameters for the sampling.

    Attributes:
        sample_data (bool):
            Flag indicating whether to sample the data.
        sample_percentage (int):
            Percentage of data to sample.
        optimum_k (str):
            Optimum value of k for clustering.
        min_cluster (int):
            Minimum number of clusters.
        max_cluster (int):
            Maximum number of clusters.
    """

    sample_data: bool = False
    sample_percentage: int = 5
    optimum_k: str = "auto"
    min_cluster: int = 2
    max_cluster: int = 30

    def __post_init__(self):
        super().__init__()
