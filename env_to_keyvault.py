"""
This script is used to create secrets in Azure Keyvault from the environment variables.

For the list of environment parameters that will be created as secrets, please refer to the Environment class in rag_experiment_accelerator/config/environment.py.
"""

import argparse

from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    environment = Environment.from_env_or_keyvault()
    logger.info("Creating secrets in Keyvault from the environment")
    logger.info("The following secrets will be created:")
    for secret in environment.fields():
        logger.info(f"  - {secret[0]}")

    environment.to_keyvault()
    logger.info(
        f"Secrets in Keyvault {environment.azure_key_vault_endpoint} have been created successfully."
    )
