from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import argparse
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import json
import os


def create_data_asset(data_path, data_asset_name):

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

    print(aml_dataset_unlabeled.version)
    print(aml_dataset_unlabeled.id)

    return aml_dataset_unlabeled.version