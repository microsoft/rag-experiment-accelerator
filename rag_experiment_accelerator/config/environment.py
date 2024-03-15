import os
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError
import openai
from typing import Iterable, Tuple

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.utils.auth import get_default_az_cred

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


def field_to_azure_key_vault_endpoint(field_name: str) -> str:
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
    azure_key_vault_endpoint: Optional[str]

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
    def _keyvault(cls, azure_key_vault_endpoint: str) -> SecretClient:
        return SecretClient(
            azure_key_vault_endpoint,
            credential=get_default_az_cred(),
        )

    @classmethod
    def _field_names(cls) -> Iterable[str]:
        return vars(cls)["__dataclass_fields__"].keys()

    @classmethod
    def _is_field_optional(cls, field_name: str) -> bool:
        return vars(cls)["__dataclass_fields__"][field_name].type == Optional[str]

    def fields(self) -> Iterable[Tuple[str, str]]:
        return vars(self).items()

    @classmethod
    def _get_value_from_keyvault(
        cls, keyvault: SecretClient, field_name: str, is_optional: bool = False
    ) -> Optional[str]:
        try:
            value = keyvault.get_secret(
                field_to_azure_key_vault_endpoint(field_name)
            ).value
            # None values are stored as 'None'
            if value == "None":
                value = None
            return value
        except ResourceNotFoundError as e:
            if is_optional:
                return None
            raise e

    @classmethod
    def _from_env(cls) -> "Environment":
        values_dict = {
            name: _get_env_var(field_to_env_name(name), cls._is_field_optional(name))
            for name in cls._field_names()
        }
        return cls(**values_dict)

    @classmethod
    def from_keyvault(cls, azure_key_vault_endpoint: str) -> "Environment":
        """
        Initialize the Environment using the keyvault endpoint provided.
        """
        keyvault = cls._keyvault(azure_key_vault_endpoint=azure_key_vault_endpoint)
        values_dict = {
            field_name: cls._get_value_from_keyvault(keyvault, field_name)
            for field_name in cls._field_names()
        }
        return cls(**values_dict)

    @classmethod
    def from_env_or_keyvault(cls) -> "Environment":
        use_key_vault = _get_env_var("USE_KEY_VAULT", is_optional=True)

        if use_key_vault and use_key_vault.lower() == "true":
            # Most values will be found in env, but secrets will be found in keyvault
            azure_key_vault_endpoint = _get_env_var(
                "AZURE_KEY_VAULT_ENDPOINT", is_optional=False
            )
            keyvault = cls._keyvault(azure_key_vault_endpoint)
            values_dict = {}

            # Try to get values from env first
            for field_name in cls._field_names():
                is_optional = cls._is_field_optional(field_name)
                value = _get_env_var(field_to_env_name(field_name), is_optional=True)
                # If not found in env, try to get from keyvault
                if not value:
                    value = cls._get_value_from_keyvault(
                        keyvault, field_name, is_optional=True
                    )
                    if not value and not is_optional:
                        raise ValueError(
                            f"Value for {field_name} not found in environment or keyvault"
                        )
                values_dict[field_name] = value
            return cls(**values_dict)

        return cls._from_env()

    def to_keyvault(self, azure_key_vault_endpoint: str = None) -> None:
        if not azure_key_vault_endpoint:
            if not self.azure_key_vault_endpoint:
                raise ValueError(
                    "Keyvault endpoint not provided and not set in .env file"
                )
            azure_key_vault_endpoint = self.azure_key_vault_endpoint
        keyvault = Environment._keyvault(
            azure_key_vault_endpoint=azure_key_vault_endpoint
        )
        for field_name, value in self.fields():
            keyvault.set_secret(
                name=field_to_azure_key_vault_endpoint(field_name), value=str(value)
            )
