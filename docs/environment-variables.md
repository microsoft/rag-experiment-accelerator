# Resource Deployment and Environment Variables

## Required Resources

To use the rag-experiment-accelerator, create the following resources:
- [Azure Search Service](https://azure.microsoft.com/en-us/products/ai-services/cognitive-search)
    - Turning on [Semantic Search](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview) is optional
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)
    - Create new deployments using models `gpt-35-turbo` and `text-embedding-ada-002`
- [Azure ML Workspace](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace?view=azureml-api-2)
- [Azure AI Service for Language](https://learn.microsoft.com/en-us/azure/search/cognitive-search-skill-language-detection) is optional



## Environment Variables

Below are the required environment variables, to be added to a local .env file at the root of the repo.
| Variable                      | Value                                      | Note                                                                     |
|-------------------------------|--------------------------------------------|--------------------------------------------------------------------------|
| AZURE_SEARCH_SERVICE_ENDPOINT |                                            | Azure Cognitive Search API Endpoint                                      |
| AZURE_SEARCH_ADMIN_KEY        |                                            | Azure Cognitive Search Key                                               |
| OPENAI_ENDPOINT               |                                            | Azure OpenAI API endpoint                                                |
| OPENAI_API_KEY                |                                            | Azure OpenAI API Key                                                     |
| OPENAI_API_TYPE               | Azure, open_ai                             | Azure, open_ai, or excluded depending on OpenAI Service being used       |
| OPENAI_API_VERSION            | 2023-03-15-preview                         | See https://learn.microsoft.com/en-us/azure/ai-services/openai/reference |
| SUBSCRIPTION_ID               |                                            | Azure subscription ID                                                    |
| WORKSPACE_NAME                |                                            | Name of deployed Azure ML Workspace                                      |
| RESOURCE_GROUP_NAME           |                                            | Azure resource group name                                                |
| LOGGING_LEVEL                 | NOTSET, DEBUG, INFO, WARN, ERROR, CRITICAL | LOGGING_LEVEL is INFO by default                                         |

Below are optional environment variables, to be added to a local .env file at the root of the repo.
| Variable                        | Value                                      | Note                                                                     |
|---------------------------------|--------------------------------------------|--------------------------------------------------------------------------|
| AZURE_LANGUAGE_SERVICE_ENDPOINT |                                            | Azure AI Service for Language API Endpoint                               |
| AZURE_LANGUAGE_SERVICE_KEY      |                                            | Azure AI Service for Language Key                                        |