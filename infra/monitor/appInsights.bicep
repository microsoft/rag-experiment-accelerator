@description('Location for all resources.')
param Location string

@description('Name of the Log Analytics workspace.')
param LogWorkspaceName string

@description('Name of the Application Insights.')
param ApplicationInsightsName string

resource LogWorkspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: LogWorkspaceName
  location: Location
  properties: {
  }
}


resource ApplicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: ApplicationInsightsName
  location: Location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: LogWorkspace.id
  }
}

output ApplicationInsightsId string = ApplicationInsights.id
