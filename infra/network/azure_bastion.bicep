param vnetName string
param bastionName string
param bastionSubnetName string
param location string
param publicIpName string // Name of the existing public IP resource

resource bastion 'Microsoft.Network/bastionHosts@2023-04-01' = {
  name: bastionName
  location: location
  properties: {
    dnsName: bastionName
    ipConfigurations: [
      {
        name: 'bastionIpConfig'
        properties: {
          subnet: {
            id: resourceId('Microsoft.Network/virtualNetworks/subnets', vnetName, bastionSubnetName)
          }
          publicIPAddress: {
            id: resourceId('Microsoft.Network/publicIPAddresses', publicIpName)
          }
        }
      }
    ]
  }
}

output bastionFqdn string = bastion.properties.dnsName
