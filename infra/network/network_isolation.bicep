param vnetName string = 'myVNet'
param location string
param vnetAddressSpace string
param proxySubnetName string
param proxySubnetAddressSpace string
param azureSubnetName string
param azureSubnetAddressSpace string

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
