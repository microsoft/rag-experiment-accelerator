param name string
param location string = resourceGroup().location
param tags object = {}

@description('Service principal will be granted read access to the KeyVault. If unset, no service principal is granted access by default')
param principalId string = ''

var defaultAccessPolicies = !empty(principalId) ? [
  {
    objectId: principalId
    permissions: { secrets: [ 'get', 'set', 'list' ] }
    tenantId: subscription().tenantId
  }
] : []

resource keyVault 'Microsoft.KeyVault/vaults@2022-07-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    tenantId: subscription().tenantId
    sku: { family: 'A', name: 'standard' }
    enabledForTemplateDeployment: true
    accessPolicies: union(defaultAccessPolicies, [
      // define access policies here
    ])
  }
}

output id string = keyVault.id
output endpoint string = keyVault.properties.vaultUri
output name string = keyVault.name
