# Promptflow Secret Setup

## Prerequisites
Ensure that your `.env` file is properly created, instructions [here](../../../README.md#installation), install the dev-requirements and login to the az cli.
``` bash
# Install the dev requirements
pip install -r dev-requirements.txt 

# Login to the az cli
az login
```

## AzureML Connections
AzureML connections are recommended as the secrets are stored securely in Key Vault and config is stored in the workspace.

Create a custom connection in the AzureML workspace, instructions [here](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/tools-reference/python-tool?view=azureml-api-2#create-a-custom-connection).

The key value pairs should correspond to the variables in the `.env.template` [here](.env.template).

The following variables are required to be set as secret:
- AZURE_SEARCH_ADMIN_KEY
- OPENAI_API_KEY
- AML_SUBSCRIPTION_ID
- AML_RESOURCE_GROUP_NAME
- AML_WORKSPACE_NAME

And the remaining variables must not be set as secret:
- AZURE_SEARCH_SERVICE_ENDPOINT
- OPENAI_API_TYPE
- OPENAI_ENDPOINT
- OPENAI_API_VERSION

The following variables are optional:
- AZURE_LANGUAGE_SERVICE_KEY - secret
- AZURE_LANGUAGE_SERVICE_ENDPOINT - non secret
- LOGGING_LEVEL - non secret

Configure promptflow to connect to AzureML, see instructions [here](https://microsoft.github.io/promptflow/how-to-guides/set-global-configs.html#azureml).
``` bash
# Set your promptflow connection provider to azureml
pf config set connection.provider=azureml
```
