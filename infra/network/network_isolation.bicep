param vnetName string
param location string

@minLength(1)
param vnetAddressSpace string
param proxySubnetName string

@minLength(1)
param proxySubnetAddressSpace string
param azureSubnetName string

@minLength(1)
param azureSubnetAddressSpace string
param resourcePrefix string
param azureResources array

resource vnet 'Microsoft.Network/virtualNetworks@2020-06-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        vnetAddressSpace
      ]
    }
    subnets: [
      {
        name: proxySubnetName
        properties: {
          addressPrefix: proxySubnetAddressSpace
        }
      }
      {
        name: azureSubnetName
        properties: {
          addressPrefix: azureSubnetAddressSpace
        }
      }
    ]
  }
}

resource privateEndpoints 'Microsoft.Network/privateEndpoints@2020-07-01' = [
  for (resource, i) in azureResources: {
    name: '${resourcePrefix}${resource.type}PrivateEndpoint'
    location: location
    properties: {
      privateLinkServiceConnections: [
        {
          name: '${resourcePrefix}${resource.type}PLSConnection'
          properties: {
            privateLinkServiceId: resource.resourceId
            groupIds: [resource.type]
          }
        }
      ]
      subnet: {
        id: '${vnet.id}/subnets/${azureSubnetName}'
      }
    }
  }
]
