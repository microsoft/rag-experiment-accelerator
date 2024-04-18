from unittest.mock import patch, MagicMock
from typing import Optional

from azure.keyvault.secrets import SecretClient
from azure.core.exceptions import ResourceNotFoundError

from rag_experiment_accelerator.config.environment import Environment


def mock_get_value_from_env_with_keyvault(
    var_name: str, is_optional: bool = False
) -> Optional[str]:
    if var_name == "USE_KEY_VAULT":
        return "True"
    elif var_name == "AZURE_KEY_VAULT_ENDPOINT":
        return "test_keyvault_endpoint"
    elif var_name == "OPENAI_API_TYPE":
        return "azure"
    elif var_name == "OPENAI_API_VERSION":
        return "test_api_version"
    elif var_name == "OPENAI_ENDPOINT":
        return "test_api_endpoint"
    elif var_name == "AML_SUBSCRIPTION_ID":
        return "test_subscription_id"
    elif var_name == "AML_WORKSPACE_NAME":
        return "test_workspace_name"
    elif var_name == "AML_RESOURCE_GROUP_NAME":
        return "test_resource_group_name"
    elif var_name == "AZURE_SEARCH_SERVICE_ENDPOINT":
        return "test_search_endpoint"
    elif var_name == "AZURE_SEARCH_USE_SEMANTIC_SEARCH":
        return "True"
    else:
        return None


def mock_get_secret_value_from_keyvault(
    keyvault: SecretClient, field_name: str, is_optional: bool = False
) -> Optional[str]:
    if field_name == "openai_api_key":
        return "test_openai_api_key"
    elif field_name == "azure_search_admin_key":
        return "test_admin_key"
    else:
        return None


def mock_get_any_value_from_keyvault(field_name: str) -> Optional[str]:
    return_value = MagicMock()
    if field_name == "openai-api-type":
        return_value.value = "azure"
    elif field_name == "openai-api-key":
        return_value.value = "test_openai_api_key"
    elif field_name == "openai-api-version":
        return_value.value = "test_api_version"
    elif field_name == "openai-endpoint":
        return_value.value = "test_api_endpoint"
    elif field_name == "azure-search-service-endpoint":
        return_value.value = "test_search_endpoint"
    elif field_name == "azure-search-use-semantic-search":
        return_value.value = "True"
    elif field_name == "azure-search-admin-key":
        return_value.value = "test_admin_key"
    elif field_name == "aml-subscription-id":
        return_value.value = "test_subscription_id"
    elif field_name == "aml-workspace-name":
        return_value.value = "test_workspace_name"
    elif field_name == "aml-resource-group-name":
        return_value.value = "test_resource_group_name"
    else:
        raise ResourceNotFoundError(f"Not found secret {field_name}")
    return return_value


@patch("rag_experiment_accelerator.config.environment.init_keyvault")
@patch(
    "rag_experiment_accelerator.config.environment._get_value_from_env",
    side_effect=mock_get_value_from_env_with_keyvault,
)
@patch(
    "rag_experiment_accelerator.config.environment._get_value_from_keyvault",
    side_effect=mock_get_secret_value_from_keyvault,
)
def test_create_environment_from_env_or_keyvault(_, __, mock_init_keyvault):
    environment = Environment.from_env_or_keyvault()
    mock_init_keyvault.return_value = MagicMock()

    assert environment.azure_search_service_endpoint == "test_search_endpoint"
    assert environment.aml_subscription_id == "test_subscription_id"
    assert environment.aml_workspace_name == "test_workspace_name"
    assert environment.aml_resource_group_name == "test_resource_group_name"
    assert environment.openai_api_version == "test_api_version"
    assert environment.openai_endpoint == "test_api_endpoint"
    assert environment.openai_api_type == "azure"

    assert environment.openai_api_key == "test_openai_api_key"
    assert environment.azure_search_admin_key == "test_admin_key"


@patch("rag_experiment_accelerator.config.environment.init_keyvault")
def test_create_environment_from_keyvault(mock_init_keyvault):
    mock_keyvault = MagicMock()
    mock_keyvault.get_secret = mock_get_any_value_from_keyvault
    mock_init_keyvault.return_value = mock_keyvault

    environment = Environment.from_keyvault("test_keyvault_endpoint")

    assert environment.azure_search_service_endpoint == "test_search_endpoint"
    assert environment.aml_subscription_id == "test_subscription_id"
    assert environment.aml_workspace_name == "test_workspace_name"
    assert environment.aml_resource_group_name == "test_resource_group_name"
    assert environment.openai_api_version == "test_api_version"
    assert environment.openai_endpoint == "test_api_endpoint"
    assert environment.openai_api_type == "azure"

    assert environment.openai_api_key == "test_openai_api_key"
    assert environment.azure_search_admin_key == "test_admin_key"


@patch("rag_experiment_accelerator.config.environment.init_keyvault")
def test_to_keyvault(mock_init_keyvault):
    mock_keyvault = MagicMock()
    mock_keyvault.set_secret = MagicMock()
    mock_init_keyvault.return_value = mock_keyvault

    environment = Environment(
        openai_api_type="azure",
        openai_api_key="test_openai_api_key",
        openai_api_version="test_api_version",
        openai_endpoint="test_api_endpoint",
        aml_subscription_id="test_subscription_id",
        aml_workspace_name="test_workspace_name",
        aml_resource_group_name="test_resource_group_name",
        aml_compute_name=None,
        aml_compute_instances_number=None,
        azure_search_service_endpoint="test_search_endpoint",
        azure_search_admin_key="test_admin_key",
        azure_document_intelligence_admin_key=None,
        azure_document_intelligence_endpoint=None,
        azure_language_service_endpoint=None,
        azure_language_service_key=None,
        azure_key_vault_endpoint="test_endpoint",
        azure_search_use_semantic_search="True",
    )
    environment.to_keyvault()

    assert mock_keyvault.set_secret.call_count == 17
