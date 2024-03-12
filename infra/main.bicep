@description('Provide a 2-13 character prefix for all resources.')
param ResourcePrefix string

@description('Location for all resources.')
param Location string = resourceGroup().location

@description('Location for all resources.')
param AISearchLocation string = resourceGroup().location

@description('Name of Azure OpenAI Resource')
param OpenAIName string = '${ResourcePrefix}oai'

@description('Azure OpenAI GPT Model Deployment Name')
param OpenAIGPTModel string = 'gpt-35-turbo'

@description('Azure OpenAI GPT Model Name')
param OpenAIGPTModelName string = 'gpt-35-turbo'

@description('Azure OpenAI GPT Model Version')
param OpenAIGPTModelVersion string = '0613'

@description('Name of Azure AI Search Resource')
param AISearchName string = '${ResourcePrefix}ais'

@description('Name of Azure Storage Account Resource')
param StorageAccountName string = '${ResourcePrefix}sa'

@description('Name of Azure Key Vault Resource')
param KeyVaultName string = '${ResourcePrefix}kv'

@description('Name of Azure Application Insights Resource')
param ApplicationInsightsName string = '${ResourcePrefix}ai'

@description('Name of Azure Machine Learning Workspace')
param MachineLearningName string = '${ResourcePrefix}aml'

@description('Flag to deply network and other netwrook related resources')
param DeployResourcesWithIsolatedNetwork bool = false

@description('Address space for the virtual network')
param VnetAddressSpace string = '10.0.0.0/16'

@description('Subnet name for the proxy server subnet')
param ProxySubnetName string = 'RemoteAccessSubnet'

@description('Address space for the proxy server subnet')
param ProxySubnetAddressSpace string = '10.0.0.0/24'

@description('Subnet name for other azure resources')
param AzureSubnetName string = 'AzureResourceSubnet'

@description('Address space for the other azure resources subnet')
param AzureSubnetAddressSpace string = '10.0.1.0/24'

resource OpenAI 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: OpenAIName
  location: Location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {}

  resource OpenAIGPTDeployment 'deployments@2023-05-01' = {
    name: OpenAIGPTModelName
    properties: {
      model: {
        format: 'OpenAI'
        name: OpenAIGPTModel
        version: OpenAIGPTModelVersion
      }
    }
    sku: {
      name: 'Standard'
      capacity: 30
    }
  }
}

// https://aka.ms/semanticsearchavailability for list of available regions.
resource AISearch 'Microsoft.Search/searchServices@2023-11-01' = {
  name: AISearchName
  location: AISearchLocation
  sku: {
    name: 'standard'
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    semanticSearch: 'free'
  }
}

resource StorageAccount 'Microsoft.Storage/storageAccounts@2021-08-01' = {
  name: StorageAccountName
  location: Location
  kind: 'StorageV2'
  sku: {
    name: 'Standard_GRS'
  }
}

resource KeyVault 'Microsoft.KeyVault/vaults@2021-11-01-preview' = {
  name: KeyVaultName
  location: Location
  properties: {
    sku: {
      name: 'standard'
      family: 'A'
    }
    tenantId: subscription().tenantId
    accessPolicies: []
  }
}

module monitoring 'monitor/appInsights.bicep' = {
  name: 'appInsights'
  params: {
    ApplicationInsightsName: ApplicationInsightsName
    Location: Location
    LogWorkspaceName: '${ResourcePrefix}lw'
  }
}

resource MachineLearningWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-06-01-preview' = {
  name: MachineLearningName
  location: Location 
  identity: {
    type: 'systemAssigned'
  }
  properties: {
    storageAccount: StorageAccount.id
    keyVault: KeyVault.id
    applicationInsights: monitoring.outputs.ApplicationInsightsId
  }
}

module network_resources 'network/network_isolation.bicep' = if (DeployResourcesWithIsolatedNetwork) {
  name: 'network_isolation_resources'
  params: {
    location: Location
    vnetAddressSpace: VnetAddressSpace
    proxySubnetName: ProxySubnetName
    proxySubnetAddressSpace: ProxySubnetAddressSpace
    azureSubnetName: AzureSubnetName
    azureSubnetAddressSpace: AzureSubnetAddressSpace
  }
}
