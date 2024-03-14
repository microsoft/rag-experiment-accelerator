"""
This script is used to create secrets in Azure Keyvault from the environment variables.

For the list of environment parameters that will be created as secrets, please refer to the Environment class in rag_experiment_accelerator/config/environment.py.

Noe that this script will create secrets with the value None for those environment parameters that are of type Optional[str].
"""

import argparse

from rag_experiment_accelerator.config.environment import Environment
from rag_experiment_accelerator.utils.logging import get_logger


logger = get_logger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="path to the config file",
        default="./config.json",
    )
    parser.add_argument(
        "--keyvault_name",
        type=str,
        help="keyvault name to create secrets in, if not provided, will attempt to read from .env file",
        default=None,
    )
    args, _ = parser.parse_known_args()

    environment = Environment.from_env()
    logger.info("Creating secrets in Keyvault from the environment")
    logger.info("The following secrets will be created:")
    for secret in environment.fields():
        logger.info(f"  - {secret[0]}")

    environment.to_keyvault()
    logger.info(
        f"Secrets in Keyvault {environment.keyvault_name} have been created successfully."
    )
