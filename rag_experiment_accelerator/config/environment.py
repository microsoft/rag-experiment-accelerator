import os
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
import openai
from typing import Iterable, Tuple

from rag_experiment_accelerator.utils.logging import get_logger

load_dotenv(override=True)
logger = get_logger(__name__)


def _get_env_var(var_name: str, is_optional: bool = False) -> Optional[str]:
    """Get the value of an environment variable.

    Raises ValueError if not found.
    """
    var = os.getenv(var_name, None)
    if var is None and not is_optional:
        logger.critical(f"{var_name} environment variable not set.")
        raise ValueError(f"{var_name} environment variable not set.")
    return var


def field_to_keyvault_name(field_name: str) -> str:
    return field_name.replace("_", "-")


def field_to_env_name(field_name: str) -> str:
    return field_name.upper()


@dataclass
class Environment:
    openai_api_type: str
    openai_api_key: str
    openai_api_version: str
    openai_endpoint: str
    aml_subscription_id: str
    aml_workspace_name: str
    aml_resource_group_name: str
    aml_compute_name: Optional[str]
    aml_compute_instances_number: Optional[str]
    azure_search_service_endpoint: str
    azure_search_admin_key: str
    azure_language_service_endpoint: Optional[str]
    azure_language_service_key: Optional[str]
    azure_document_intelligence_endpoint: Optional[str]
    azure_document_intelligence_admin_key: Optional[str]
    keyvault_name: Optional[str]

    def __post_init__(self) -> None:
        """Checks consistency of the environment settigns and sets any credentials.

        Raises:
            ValueError: If openai_api_type is not 'azure' or 'open_ai'.
        """
        if self.openai_api_type is not None and self.openai_api_type not in [
            "azure",
            "open_ai",
        ]:
            logger.critical("OPENAI_API_TYPE must be either 'azure' or 'open_ai'.")
            raise ValueError("OPENAI_API_TYPE must be either 'azure' or 'open_ai'.")
        self._set_openai_credentials()

    def _set_openai_credentials(self) -> None:
        """Sets the OpenAI credentials."""
        if self.openai_api_type is not None:
            openai.api_type = self.openai_api_type
            openai.api_key = self.openai_api_key

            if self.openai_api_type == "azure":
                openai.api_version = self.openai_api_version
                openai.api_base = self.openai_endpoint

    @classmethod
    def _keyvault(cls, keyvault_name: str) -> SecretClient:
        return SecretClient(
            f"https://{keyvault_name}.vault.azure.net/",
            credential=DefaultAzureCredential(),
        )

    @classmethod
    def _field_names(cls) -> Iterable[str]:
        return vars(cls)["__dataclass_fields__"].keys()

    @classmethod
    def _is_field_optional(cls, field_name: str) -> bool:
        return vars(cls)["__dataclass_fields__"][field_name].type == Optional[str]

    def _fields(self) -> Iterable[Tuple[str, str]]:
        return vars(self).items()

    @classmethod
    def _get_value_from_keyvault(
        cls, keyvault: SecretClient, field_name: str
    ) -> Optional[str]:
        try:
            value = keyvault.get_secret(field_to_keyvault_name(field_name)).value
            # None values are stored as 'None'
            if value == "None":
                value = None
            return value
        except ResourceNotFoundError as e:
            if cls._is_field_optional(field_name):
                return None
            raise e

    @classmethod
    def from_env(cls) -> "Environment":
        values_dict = {
            name: _get_env_var(field_to_env_name(name), cls._is_field_optional(name))
            for name in cls._field_names()
        }
        return cls(**values_dict)

    @classmethod
    def from_keyvault(cls, keyvault_name: str) -> "Environment":
        keyvault = cls._keyvault(keyvault_name=keyvault_name)
        values_dict = {
            field_name: cls._get_value_from_keyvault(keyvault, field_name)
            for field_name in cls._field_names()
        }
        return cls(**values_dict)

    def to_keyvault(self) -> None:
        keyvault = Environment._keyvault(keyvault_name=self.keyvault_name)
        # We check if the secret is already there and has the same value,
        # to avoid creating many versions of the same secret
        for field_name, value in self._fields():
            previous_value = Environment._get_value_from_keyvault(keyvault, field_name)
            if previous_value != str(value):
                keyvault.set_secret(
                    name=field_to_keyvault_name(field_name), value=str(value)
                )
