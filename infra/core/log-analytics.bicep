param name string
param location string = resourceGroup().location
param tags object = {}

param skuName string = 'PerGB2018'
param retentionInDays int = 30
param dailyQuotaGb int = 1

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: name
  location: location
  tags: tags
  properties: {
    sku: {
      name: skuName
    }
    retentionInDays: retentionInDays
    workspaceCapping: {
      dailyQuotaGb: dailyQuotaGb
    }
  }
}

output id string = logAnalytics.id
output name string = logAnalytics.name