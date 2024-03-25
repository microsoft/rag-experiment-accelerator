param keyVaultName string = ''
param azureOpenAIName string = ''
param azureAISearchName string = ''
param rgName string = ''
// Do not use _ in the key names as it is not allowed in the key vault secret name
param openAIKeyName string = 'openai-api-key'
param searchKeyName string = 'azure-search-admin-key'

resource openAIKeySecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  parent: keyVault
  name: openAIKeyName
  properties: {
    contentType: 'string'
    value: listKeys(resourceId(subscription().subscriptionId, rgName, 'Microsoft.CognitiveServices/accounts', azureOpenAIName), '2023-05-01').key1
  }
}

resource searchKeySecret 'Microsoft.KeyVault/vaults/secrets@2022-07-01' = {
  parent: keyVault
  name: searchKeyName
  properties: {
    contentType: 'string'
    value: listAdminKeys(resourceId(subscription().subscriptionId, rgName, 'Microsoft.Search/searchServices', azureAISearchName), '2023-11-01').primaryKey
}
}

resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' existing = {
  name: keyVaultName
}

output SEARCH_KEY_NAME string = searchKeySecret.name
output OPENAI_KEY_NAME string = openAIKeySecret.name
