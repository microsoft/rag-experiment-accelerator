param Location string = 'uksouth'
param DeployResourcesWithIsolatedNetwork bool = true

module network_resources 'network/network_isolation.bicep' = if (DeployResourcesWithIsolatedNetwork) {
  name: 'network_isolation_resources'
  params: {
    location: Location
  }
}
