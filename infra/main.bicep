@description('Location for all resources')
param location string = resourceGroup().location
@description('Base application name')
param appName string
@description('Web App SKU (App Service Plan)')
param appServiceSku string = 'B1'
@description('ACR SKU')
param acrSku string = 'Basic'
@description('Container image tag to deploy')
param imageTag string = 'latest'

var acrName = toLower(replace('${appName}acr','-',''))
var storageName = toLower(replace(replace(uniqueString(resourceGroup().id, appName, 'stg'), '-', ''), '_', ''))
var workspaceName = '${appName}-log'
var insightsName = '${appName}-ai'
var planName = '${appName}-plan'
var imageRepo = 'api'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: workspaceName
  location: location
  properties: {
    retentionInDays: 30
    features: { legacy: 0 }
  }
}

resource appInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: insightsName
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    Flow_Type: 'Redfield'
    WorkspaceResourceId: logAnalytics.id
  }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: storageName
  location: location
  sku: { name: 'Standard_LRS' }
  kind: 'StorageV2'
  properties: {
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true
  }
}

resource artifactsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  name: '${storage.name}/default/artifacts'
  properties: {}
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' = {
  name: acrName
  location: location
  sku: { name: acrSku }
  properties: { adminUserEnabled: false }
}

resource plan 'Microsoft.Web/serverfarms@2023-01-01' = {
  name: planName
  location: location
  sku: {
    name: appServiceSku
    tier: (appServiceSku == 'B1') ? 'Basic' : 'PremiumV3'
  }
  kind: 'linux'
  properties: { reserved: true }
}

resource web 'Microsoft.Web/sites@2023-01-01' = {
  name: appName
  location: location
  kind: 'app,linux,container'
  identity: { type: 'SystemAssigned' }
  properties: {
    serverFarmId: plan.id
    siteConfig: {
      linuxFxVersion: 'DOCKER|${acr.properties.loginServer}/${imageRepo}:${imageTag}'
      appSettings: [
        { name: 'WEBSITES_PORT'; value: '8000' }
        { name: 'USE_ONNX'; value: '1' }
        { name: 'ENABLE_BLOB_UPLOAD'; value: '1' }
        { name: 'STORAGE_CONTAINER'; value: 'artifacts' }
        { name: 'BLOB_CONTAINER'; value: 'im2fit-outputs' }
        { name: 'APP_ENV'; value: 'demo' }
        { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING'; value: appInsights.properties.ConnectionString }
      ]
      alwaysOn: true
      http20Enabled: true
      ftpsState: 'Disabled'
    }
    httpsOnly: true
  }
  dependsOn: [acr]
}

resource acrPullRA 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(web.id, 'AcrPull')
  scope: acr
  properties: {
    principalId: web.identity.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
    principalType: 'ServicePrincipal'
  }
}

resource storageBlobRA 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  name: guid(web.id, 'StorageBlobDataContributor')
  scope: storage
  properties: {
    principalId: web.identity.principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe')
    principalType: 'ServicePrincipal'
  }
}

output webAppUrl string = 'https://${web.name}.azurewebsites.net'
output appInsightsConnectionString string = appInsights.properties.ConnectionString
output storageAccountName string = storage.name
output acrLoginServer string = acr.properties.loginServer
