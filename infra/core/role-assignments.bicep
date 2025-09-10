param principalId string
param storageAccountName string  
param containerRegistryName string

// Built-in Azure role definition IDs
var roles = {
  StorageBlobDataContributor: 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'
  AcrPull: '7f951dda-4ed3-4680-a7ca-43fe172d538d'
}

resource storageAccount 'Microsoft.Storage/storageAccounts@2022-05-01' existing = {
  name: storageAccountName
}

resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-01-01-preview' existing = {
  name: containerRegistryName
}

// Assign Storage Blob Data Contributor role for blob access
resource storageRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: storageAccount
  name: guid(storageAccount.id, principalId, roles.StorageBlobDataContributor)
  properties: {
    principalId: principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.StorageBlobDataContributor)
    principalType: 'ServicePrincipal'
  }
}

// Assign ACR Pull role for container registry access
resource acrRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: containerRegistry
  name: guid(containerRegistry.id, principalId, roles.AcrPull)
  properties: {
    principalId: principalId
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', roles.AcrPull)
    principalType: 'ServicePrincipal'
  }
}