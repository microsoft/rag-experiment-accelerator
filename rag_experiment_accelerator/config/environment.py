import os
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError
from typing import Tuple

from rag_experiment_accelerator.utils.logging import get_logger
from rag_experiment_accelerator.utils.auth import get_default_az_cred

load_dotenv(override=True)
logger = get_logger(__name__)


def field_to_env_name(field_name: str) -> str:
    """
    Convert from the name of a field to an environment variable name.
    For example, openai_api_key becomes OPENAI_API_KEY.
    """
    return field_name.upper()


def _get_value_from_env(var_name: str, is_optional: bool = False) -> Optional[str]:
    """
    Get the value of an environment variable.

    Raises ValueError if not found and is not optional.
    """
    var = os.getenv(var_name, None)
    if var is None and not is_optional:
        logger.critical(f"{var_name} environment variable not set.")
        raise ValueError(f"{var_name} environment variable not set.")
    return var


def init_keyvault(azure_key_vault_endpoint: str) -> SecretClient:
    """
    Initializes keyvault client using the provided endpoint and default credentials.
    """
    return SecretClient(
        azure_key_vault_endpoint,
        credential=get_default_az_cred(),
    )


def field_to_keyvault_name(field_name: str) -> str:
    """
    Convert from the name of a field to a keyvault secret name.
    For example, openai_api_key becomes openai-api-key.
    """
    return field_name.replace("_", "-")


def _get_value_from_keyvault(
    keyvault: SecretClient, field_name: str, is_optional: bool = False
) -> Optional[str]:
    """
    Retrieves the value from the provided keyvault.

    Raises ResourceNotFoundError if not found and is not optional.
    """
    try:
        value = keyvault.get_secret(field_to_keyvault_name(field_name)).value
        # None values are stored as 'None'
        if value == "None":
            value = None
        return value
    except ResourceNotFoundError as e:
        if is_optional:
            return None
        raise e


@dataclass
class Environment:
    openai_api_type: Optional[str]
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
    azure_search_use_semantic_search: str
    azure_language_service_endpoint: Optional[str]
    azure_language_service_key: Optional[str]
    azure_document_intelligence_endpoint: Optional[str]
    azure_document_intelligence_admin_key: Optional[str]
    azure_key_vault_endpoint: Optional[str]

    @classmethod
    def _field_names(cls) -> list[str]:
        """
        Returns a list of all field names of this class
        """
        return list(vars(cls)["__dataclass_fields__"].keys())

    @classmethod
    def _is_field_optional(cls, field_name: str) -> bool:
        """
        Returns whether a field is optional based on it's type
        Fields with type Optional[str] are optional, fields with type str are required
        """
        return vars(cls)["__dataclass_fields__"][field_name].type == Optional[str]

    def fields(self) -> list[Tuple[str, str]]:
        """
        Returns a list of tuples containing the field name and value of this class instance
        """
        return list(vars(self).items())

    @classmethod
    def _from_env(cls) -> "Environment":
        """
        Initialize the Environment using the environment variables.
        """
        values_dict = {
            name: _get_value_from_env(
                field_to_env_name(name), cls._is_field_optional(name)
            )
            for name in cls._field_names()
        }
        return cls(**values_dict)

    @classmethod
    def from_keyvault(cls, azure_key_vault_endpoint: str) -> "Environment":
        """
        Initialize the Environment using the keyvault endpoint provided.
        """
        keyvault = init_keyvault(azure_key_vault_endpoint=azure_key_vault_endpoint)
        values_dict = {
            field_name: _get_value_from_keyvault(
                keyvault, field_name, cls._is_field_optional(field_name)
            )
            for field_name in cls._field_names()
        }
        return cls(**values_dict)

    @classmethod
    def from_env_or_keyvault(cls) -> "Environment":
        """
        Initialize the Environment using the environment variables and keyvault.

        If USE_KEY_VAULT is set to True, this will use environment variables for those values that are set there.
        For those values that are not set in the environment, it will attempt to use the keyvault.

        If USE_KEY_VAULT is not set to True, this will use the environment variables only.

        Note that this method won't work from within AzureML compute, in that case you need to use from_keyvault().

        Raises:
            ValueError: If a required value is not found in the environment or keyvault.
        """
        use_key_vault = _get_value_from_env("USE_KEY_VAULT", is_optional=True)

        if use_key_vault and use_key_vault.lower() == "true":
            # Most values will be found in env, but secrets will be found in keyvault
            azure_key_vault_endpoint = _get_value_from_env(
                "AZURE_KEY_VAULT_ENDPOINT", is_optional=False
            )
            keyvault = init_keyvault(azure_key_vault_endpoint)
            values_dict = {}

            # Try to get values from env first
            for field_name in cls._field_names():
                is_optional = cls._is_field_optional(field_name)
                value = _get_value_from_env(
                    field_to_env_name(field_name), is_optional=True
                )
                # If not found in env, try to get from keyvault
                if not value:
                    value = _get_value_from_keyvault(
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
        """
        Serializes the environment to keyvault.
        Note that for the optional fields that are not set, this will create the value 'None' in the keyvault.

        Raises:
            ValueError if the keyvault endpoint is not provided and not set in the environment.
        """
        if not azure_key_vault_endpoint:
            if not self.azure_key_vault_endpoint:
                raise ValueError(
                    "Keyvault endpoint not provided and not set in environment"
                )
            azure_key_vault_endpoint = self.azure_key_vault_endpoint
        keyvault = init_keyvault(azure_key_vault_endpoint=azure_key_vault_endpoint)
        for field_name, value in self.fields():
            keyvault.set_secret(
                name=field_to_keyvault_name(field_name), value=str(value)
            )
