from azure.identity import DefaultAzureCredential

from rag_experiment_accelerator.utils.logging import get_logger

logger = get_logger(__name__)


def get_default_az_cred():
    """
    Returns a DefaultAzureCredential object that can be used to authenticate with Azure services.
    If the credential cannot be obtained, an error is logged and an exception is raised.
    """
    try:
        credential = DefaultAzureCredential()
        # Check if credential can get token successfully.
        credential.get_token("https://management.azure.com/.default")
    except Exception as ex:
        logger.error(
            "Unable to get a token from DefaultAzureCredential. Please run 'az"
            " login' in your terminal and try again."
        )
        raise ex
    return credential
