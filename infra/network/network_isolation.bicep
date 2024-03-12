param vnetName string = 'myVNet'
param location string
param addressPrefix string = '10.0.0.0/16'
param subnet1Name string = 'subnet1'
param subnet1Prefix string = '10.0.0.0/24'
param subnet2Name string = 'subnet2'
param subnet2Prefix string = '10.0.1.0/24'

resource vnet 'Microsoft.Network/virtualNetworks@2020-06-01' = {
  name: vnetName
  location: location
  properties: {
    addressSpace: {
      addressPrefixes: [
        addressPrefix
      ]
    }
    subnets: [
      {
        name: subnet1Name
        properties: {
          addressPrefix: subnet1Prefix
        }
      }
      {
        name: subnet2Name
        properties: {
          addressPrefix: subnet2Prefix
        }
      }
    ]
  }
}
