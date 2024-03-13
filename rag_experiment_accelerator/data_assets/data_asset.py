from azure.ai.ml import MLClient
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.utils.auth import get_default_az_cred
from rag_experiment_accelerator.config.environment import Environment

logger = get_logger(__name__)


def create_data_asset(data_path: str, data_asset_name: str, environment: Environment):
    """
    Creates a new data asset in Azure Machine Learning workspace.

    Args:
        data_path (str): The path to the data file.
        data_asset_name (str): The name of the data asset.
        environment (Environment): Class containing the environment configuration

    Returns:
        int: The version of the created data asset.
    """

    ml_client = MLClient(
        get_default_az_cred(),
        environment.aml_subscription_id,
        environment.aml_resource_group_name,
        environment.aml_workspace_name,
    )

    aml_dataset = Data(
        path=data_path,
        type=AssetTypes.URI_FILE,
        description="rag data",
        name=data_asset_name,
    )

    ml_client.data.create_or_update(aml_dataset)

    aml_dataset_unlabeled = ml_client.data.get(name=data_asset_name, label="latest")

    logger.info(f"Dataset version: {aml_dataset_unlabeled.version}")
    logger.info(f"Dataset ID: {aml_dataset_unlabeled.id}")

    return aml_dataset_unlabeled.version
