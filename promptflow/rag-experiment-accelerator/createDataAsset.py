from promptflow import tool
from promptflow.connections import CustomConnection
from azure.ai.ml import command, Input, MLClient, UserIdentityConfiguration, ManagedIdentityConfiguration
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes, InputOutputModes
from azure.identity import DefaultAzureCredential
# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def createFileDataAsset(previousStepCompleted: bool, conn: CustomConnection, path: str, description:str, name: str) -> bool:

    subscription_id = conn.azure_sub
    resource_group = conn.rg
    workspace = conn.aml_workspace_name

    # connect to the AzureML workspace
    ml_client = MLClient(DefaultAzureCredential(), subscription_id,resource_group,workspace)

    # ==============================================================
    # What type of data does the path point to? Options include:
    # data_type = AssetTypes.URI_FILE # a specific file
    # data_type = AssetTypes.URI_FOLDER # a folder
    # ==============================================================
    
    my_data = Data(path,description,name,type=AssetTypes.URI_FILE)

    # Create the data asset in the workspace
    ml_client.data.create_or_update(my_data)

    return True