# **RAG Experiment Accelerator** with Prompt Flow

# Folder structure
# Documentation
# What you will learn
# Prerequisites
- [Azure Cognitive Search Service](https://learn.microsoft.com/en-us/azure/search/search-create-service-portal) (Note: [Semantic Search](https://learn.microsoft.com/en-us/azure/search/search-get-started-semantic?tabs=dotnet) is available in Azure Cognitive Search, at Basic tier or higher.)
- [Azure OpenAI Service](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview#how-do-i-get-access-to-azure-openai) or access to the [OpenAI API](https://platform.openai.com/docs/quickstart?context=python)
- [Azure Machine Learning Resources](https://learn.microsoft.com/en-us/azure/machine-learning/tutorial-azure-ml-in-a-day?view=azureml-api-2)

## Promptflow Secret Setup
Install the dev-requirements and login to the az cli.
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
## Getting Started
To use the **RAG Experiment Accelerator** in Prompt Flow, follow these steps:
- 1. Modify the `config.json` file with the hyperparameters for your experiment. Full documentation on can be found [here](../README.md#description-of-configuration-elements)
- 2. Setup connection [add setup instruction here]

## Creating a custom environment for prompt flow runtime

```bash
az login

az account set --subscription <subscription ID>

az extension add --name ml

az configure --defaults workspace=$MLWorkSpaceName group=$ResourceGroupName

cd ./promptflow/rag-experiment-accelerator/custom_environment

az ml environment create --file ./environment.yaml -w $MLWorkSpaceName
```