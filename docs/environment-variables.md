# Resource Deployment and Environment Variables

## Required Resources

To use the rag-experiment-accelerator, create the following resources:
- [Azure Search Service](https://azure.microsoft.com/en-us/products/ai-services/cognitive-search)
    - Turning on [Semantic Search](https://learn.microsoft.com/en-us/azure/search/semantic-search-overview) is optional
- [Azure OpenAI](https://learn.microsoft.com/en-us/azure/ai-services/openai/overview)
    - Create new deployments using models `gpt-35-turbo` and `text-embedding-ada-002`
- [Azure ML Workspace](https://learn.microsoft.com/en-us/azure/machine-learning/concept-workspace?view=azureml-api-2)


## Environment Variables

Below are the required environment variables, to be added to a local .env file at the root of the repo.
| Variable                      | Value                                      | Note                                                                     |
|-------------------------------|--------------------------------------------|--------------------------------------------------------------------------|
| AZURE_SEARCH_SERVICE_ENDPOINT |                                            | Azure Cognitive Search API Endpoint                                      |
| AZURE_SEARCH_ADMIN_KEY        |                                            | Azure Cognitive Search Key                                               |
| OPENAI_API_KEY                |                                            | OpenAI API key                                                           |
| OPENAI_API_TYPE               | azure, open_ai                             | Must be `azure` for Azure OpenAI or `open_ai` for OpenAI.                |
| OPENAI_ENDPOINT               |                                            | Azure OpenAI API endpoint.                                               |
| OPENAI_API_VERSION            | 2023-03-15-preview                         | Azure OpenAI API version. See https://learn.microsoft.com/en-us/azure/ai-services/openai/reference. |
| AML_SUBSCRIPTION_ID           |                                            | Azure Machine Learning subscription ID                                   |
| AML_WORKSPACE_NAME            |                                            | Name of deployed Azure Machine Learning Workspace                        |
| AML_RESOURCE_GROUP_NAME       |                                            | Azure Machine Learning resource group name                               |
| LOGGING_LEVEL                 | NOTSET, DEBUG, INFO, WARN, ERROR, CRITICAL | LOGGING_LEVEL is INFO by default                                         |