metadata description = 'Creates an Azure Machine Learning Workspace.'
param name string
param location string = resourceGroup().location
param tags object = {}
param storageAccount string
param keyVault string
param applicationInsights string

resource machineLearningWorkspace 'Microsoft.MachineLearningServices/workspaces@2023-06-01-preview' = {
  name: name
  location: location 
  identity: {
    type: 'systemAssigned'
  }
  tags: tags
  properties: {
    storageAccount: storageAccount
    keyVault: keyVault
    applicationInsights: applicationInsights
  }
}

output workspaceName string = machineLearningWorkspace.name
output workspaceId string = machineLearningWorkspace.id
