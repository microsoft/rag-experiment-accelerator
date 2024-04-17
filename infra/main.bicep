targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

param resourceToken string = toLower(uniqueString(subscription().id, environmentName, location))

@description('Location for all resources.')
param location string

@description('Location for all resources. https://aka.ms/semanticsearchavailability for list of available regions.')
@allowed([
  'australiaeast'
  'australiasoutheast'
  'brazilsouth'
  'canadacentral'
  'canadaeast'
  'centralindia'
  'centralus'
  'centraluseuap'
  'eastasia'
  'eastus'
  'eastus2'
  'eastus2euap'
  'eastusstg'
  'francecentral'
  'japaneast'
  'japanwest'
  'koreacentral'
  'koreasouth'
  'northcentralus'
  'northeurope'
  'qatarcentral'
  'southcentralus'
  'southeastasia'
  'switzerlandnorth'
  'uksouth'
  'ukwest'
  'westcentralus'
  'westeurope'
  'westus'
  'westus2'
  'westus3'
])
param azureAISearchLocation string = location

@description('Azure AI Search Resource')
param azureAISearchName string = '${environmentName}-search-${resourceToken}'

@description('The SKU of the search service you want to create. E.g. free or standard')
@allowed([
  'free'
  'basic'
  'standard'
  'standard2'
  'standard3'
])
param azureSearchSku string = 'standard'

@description('Name of Azure OpenAI Resource')
param azureOpenAIResourceName string = '${environmentName}-openai-${resourceToken}'

@description('Name of Azure OpenAI Resource SKU')
param azureOpenAISkuName string = 'S0'

@description('Azure OpenAI GPT Model Deployment Name')
param azureOpenAIModel string = 'gpt-35-turbo'

@description('Azure OpenAI GPT Model Name')
param azureOpenAIModelName string = 'gpt-35-turbo'

@description('Azure OpenAI GPT Model Version')
param azureOpenAIModelVersion string = '0613'

@description('Name of Azure Application Insights Resource')
param applicationInsightsName string = '${environmentName}-appinsights-${resourceToken}'

@description('Name of Storage Account')
param storageAccountName string = 'str${resourceToken}'

@description('Name of Azure Machine Learning Workspace')
param machineLearningName string = '${environmentName}-aml-${resourceToken}'

@description('Id of the user or app to assign application roles')
param principalId string = ''

var tags = { 'azd-env-name': environmentName }
var rgName = 'rg-${environmentName}'
var keyVaultName = 'kv-${resourceToken}'

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: rgName
  location: location
  tags: tags
}

// Store secrets in a keyvault
module keyvault './shared/keyvault.bicep' = {
  name: 'keyvault'
  scope: rg
  params: {
    name: keyVaultName
    location: location
    tags: tags
    principalId: principalId
  }
}

module search 'shared/search-services.bicep' = {
  name: azureAISearchName
  scope: rg
  params: {
    name: azureAISearchName
    location: azureAISearchLocation
    sku: {
      name: azureSearchSku
    }
    semanticSearch: 'free'
    tags: tags
  }
}

module openai './shared/cognitiveservices.bicep' = {
  name: azureOpenAIResourceName
  scope: rg
  params: {
    name: azureOpenAIResourceName
    location: location
    tags: tags
    sku: {
      name: azureOpenAISkuName
    }
    deployments: [
      {
        name: azureOpenAIModel
        model: {
          format: 'OpenAI'
          name: azureOpenAIModelName
          version: azureOpenAIModelVersion
        }
        sku: {
          name: 'Standard'
          capacity: 30
        }
      }
    ]
  }
}

module storage './shared/storage.bicep' = {
  name: storageAccountName
  scope: rg
  params: {
    name: storageAccountName
    location: location
    sku: {
      name: 'Standard_GRS'
    }
  }
}

module monitoring 'shared/monitoring.bicep' = {
  name: 'monitoring'
  scope: rg
  params: {
    applicationInsightsName: applicationInsightsName
    location: location
    logAnalyticsName: '${environmentName}-logAnalytics-${resourceToken}'
  }
}

module machineLearning './shared/machineLearning.bicep' = {
  name: machineLearningName
  scope: rg
  params: {
    name: machineLearningName
    location: location
    storageAccount: storage.outputs.id
    keyVault: keyvault.outputs.id
    applicationInsights: monitoring.outputs.applicationInsightsId
  }
}

module storekeys './shared/storekeys.bicep' = {
  name: 'storekeys'
  scope: rg
  params: {
    keyVaultName: keyvault.outputs.name
    azureOpenAIName: openai.outputs.name
    azureAISearchName: search.outputs.name
    rgName: rgName
  }
}

output USE_KEY_VAULT string = 'true'
output AZURE_KEY_VAULT_ENDPOINT string = keyvault.outputs.endpoint
output AZURE_SEARCH_SERVICE_ENDPOINT string = search.outputs.endpoint
output OPENAI_API_TYPE string = 'azure'
output OPENAI_ENDPOINT string = openai.outputs.endpoint
output OPENAI_API_VERSION string = '2023-03-15-preview'
output AML_SUBSCRIPTION_ID string = subscription().subscriptionId
output AML_WORKSPACE_NAME string = machineLearning.outputs.workspaceName  
output AML_RESOURCE_GROUP_NAME string = rgName
// output AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT string = 
// output AZURE_DOCUMENT_INTELLIGENCE_ADMIN_KEY string =
// output AZURE_LANGUAGE_SERVICE_ENDPOINT string =
// output AZURE_LANGUAGE_SERVICE_KEY string =
