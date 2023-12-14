# Promptflow Secret Setup

## Prerequisites
Install the dev-requirements and login to the az cli.
``` bash
# Install the dev requirements
pip install -r dev-requirements.txt 

# Login to the az cli
az login
```

## AzureML Connections
Connections are objects stored in the AzureML workspace that storage and manage credentials required for interacting with LLMs. We will be using a Custom Connection, which is a generic connection type. Custom Connections have two dictionaries, `secrets` for secrets to be stored in Key Vault, and `configs` for non secrets that are stored in the AzureML workspace. 

Create a custom connection in the AzureML workspace, instructions [here](https://learn.microsoft.com/en-us/azure/machine-learning/prompt-flow/tools-reference/python-tool?view=azureml-api-2#create-a-custom-connection). The key value pairs required are listed below.

The following variables are required to be set as secret:
- AZURE_SEARCH_ADMIN_KEY
- OPENAI_API_KEY
- AML_SUBSCRIPTION_ID
- AML_RESOURCE_GROUP_NAME
- AML_WORKSPACE_NAME

And the remaining variables must not be set as secret:
- AZURE_SEARCH_SERVICE_ENDPOINT
- OPENAI_API_TYPE - must be `azure`
- OPENAI_ENDPOINT
- OPENAI_API_VERSION

The following variables are optional:
- AZURE_LANGUAGE_SERVICE_KEY - secret
- AZURE_LANGUAGE_SERVICE_ENDPOINT - non secret
- LOGGING_LEVEL - non secret

## Configuring your connection locally 
Configure promptflow to connect to AzureML by updating the given `./azureml/config.json` with the `workspace_name`, `resource_group`, and `subscription_id` that your connection is stored in. For more information, the documentation is [here](https://microsoft.github.io/promptflow/how-to-guides/set-global-configs.html#azureml).

Update the local promptflow connection provider to look for AzureML connections. 
``` bash
# Set your promptflow connection provider to azureml
pf config set connection.provider=azureml

# Verify that the connection appears
pf connection list
```
Note: Depending on the context you're running the `pf` commands from, you may need to move the `.azureml` folder into the root of the repository.
