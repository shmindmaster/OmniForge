// Azure Infrastructure for im2fit
targetScope = 'subscription'

@minLength(1)
@maxLength(64)
@description('Name of the environment which is used to generate a short unique hash used in all resources.')
param environmentName string

@minLength(1)
@description('Primary location for all resources')
param location string

@description('Id of the user or app to assign application roles')
param principalId string = ''

// Optional parameters with defaults
@description('The image name for the container')  
param imageName string = ''

// Variables
var abbrs = loadJsonContent('abbreviations.json')
var resourceToken = toLower(uniqueString(subscription().id, environmentName, location))
var tags = {
  'azd-env-name': environmentName
  project: 'im2fit'
  purpose: 'demo'
}

// Organize resources in a resource group
resource rg 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: '${abbrs.resourcesResourceGroups}${environmentName}'
  location: location
  tags: tags
}

// Container registry for storing images
module registry './core/registry.bicep' = {
  name: 'registry'
  scope: rg
  params: {
    name: '${abbrs.containerRegistryRegistries}${resourceToken}'
    location: location
    tags: tags
  }
}

// Storage account for artifacts (blobs, files)
module storage './core/storage.bicep' = {
  name: 'storage'
  scope: rg
  params: {
    name: '${abbrs.storageStorageAccounts}${resourceToken}'
    location: location
    tags: tags
    containers: [
      {
        name: 'im2fit-outputs'
        publicAccess: 'Blob'
      }
    ]
  }
}

// Log analytics workspace
module logAnalytics './core/log-analytics.bicep' = {
  name: 'log-analytics'
  scope: rg
  params: {
    name: '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
    location: location
    tags: tags
  }
}

// Application Insights for monitoring
module appInsights './core/app-insights.bicep' = {
  name: 'app-insights'
  scope: rg
  params: {
    name: '${abbrs.insightsComponents}${resourceToken}'
    location: location
    tags: tags
    logAnalyticsWorkspaceId: logAnalytics.outputs.id
  }
}

// Container Apps environment
module containerApps './core/container-apps.bicep' = {
  name: 'container-apps'
  scope: rg
  params: {
    name: '${abbrs.appManagedEnvironments}${resourceToken}'
    location: location
    tags: tags
    logAnalyticsWorkspaceName: logAnalytics.outputs.name
  }
}

// The main application container app
module web './app/web.bicep' = {
  name: 'web'
  scope: rg
  params: {
    name: '${abbrs.appContainerApps}web-${resourceToken}'
    location: location
    tags: tags
    imageName: imageName
    containerAppsEnvironmentName: containerApps.outputs.name
    containerRegistryName: registry.outputs.name
    storageAccountName: storage.outputs.name
    applicationInsightsName: appInsights.outputs.name
    exists: false
  }
}

// Assign roles for managed identity access
module webIdentityRoleAssignments './core/role-assignments.bicep' = {
  name: 'web-identity-role-assignments'
  scope: rg
  params: {
    principalId: web.outputs.identityPrincipalId
    storageAccountName: storage.outputs.name
    containerRegistryName: registry.outputs.name
  }
}

// Optional: Assign roles for user/service principal
module userRoleAssignments './core/role-assignments.bicep' = if (!empty(principalId)) {
  name: 'user-role-assignments'  
  scope: rg
  params: {
    principalId: principalId
    storageAccountName: storage.outputs.name
    containerRegistryName: registry.outputs.name
  }
}

// Outputs
output AZURE_LOCATION string = location
output AZURE_TENANT_ID string = tenant().tenantId
output AZURE_RESOURCE_GROUP string = rg.name

output AZURE_CONTAINER_REGISTRY_ENDPOINT string = registry.outputs.loginServer
output AZURE_CONTAINER_REGISTRY_NAME string = registry.outputs.name

output AZURE_STORAGE_ACCOUNT_NAME string = storage.outputs.name
output AZURE_STORAGE_CONNECTION_STRING string = storage.outputs.connectionString

output SERVICE_WEB_IDENTITY_PRINCIPAL_ID string = web.outputs.identityPrincipalId
output SERVICE_WEB_NAME string = web.outputs.name
output SERVICE_WEB_URI string = web.outputs.uri