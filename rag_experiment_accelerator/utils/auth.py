from azure.identity import DefaultAzureCredential
from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)

def get_default_az_cred():
    try:
        credential = DefaultAzureCredential()
        # Check if given credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        logger.error("Unable to get a token from DefaultAzureCredential. Please run 'az login' in your terminal and try again.")
        raise ex