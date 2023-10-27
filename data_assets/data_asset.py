from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import os

from utils.logging import get_logger
logger = get_logger(__name__)

def create_data_asset(data_path, data_asset_name):
    """
    Creates a new data asset in Azure Machine Learning workspace.

    Args:
        data_path (str): The path to the data file.
        data_asset_name (str): The name of the data asset.

    Returns:
        int: The version of the created data asset.
    """

    ml_client = MLClient(
        DefaultAzureCredential(), os.environ['SUBSCRIPTION_ID'],os.environ['RESOURCE_GROUP_NAME'], os.environ['WORKSPACE_NAME']
    )
            
    aml_dataset = Data(
        path=data_path,
        type=AssetTypes.URI_FILE,
        description="rag data",
        name=data_asset_name,
    )

    data_create = ml_client.data.create_or_update(aml_dataset)

    aml_dataset_unlabeled = ml_client.data.get(name=data_asset_name, label="latest")

    logger.info(f"Dataset version: {aml_dataset_unlabeled.version}")
    logger.info(f"Dataset ID: {aml_dataset_unlabeled.id}")

    return aml_dataset_unlabeled.version