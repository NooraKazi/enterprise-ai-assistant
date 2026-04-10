targetScope = 'resourceGroup'

// Usage modes:
// 1) Persistent environment: keep the default accountName and deploy normally.
// 2) Restore deleted account: keep the same accountName and pass restore=true.
// 3) Throwaway environment: pass a unique accountName for each temporary deployment.
// az deployment group create --resource-group rg-enterprise-ai-assistant --template-file .\infra\azure-openai-setup.bicep --parameters accountName=ai-enterprise-ai-assistant-dev01

@description('Azure region for the Azure OpenAI account.')
param location string = resourceGroup().location

@description('Name of the Azure OpenAI account. Use a unique name if you want a fresh account instead of restoring an old one.')
@minLength(2)
@maxLength(64)
param accountName string = 'ai-enterprise-ai-assistant'

@description('Set to true only when the account name exists in soft-deleted state and you want to restore it.')
param restore bool = false

@description('Controls whether a model deployment should be created together with the account. This is skipped automatically when restore=true.')
param deployModel bool = true

@description('Deployment name inside the Azure OpenAI account.')
param deploymentName string = 'gpt-4.1-nano'

@description('Azure OpenAI model name to deploy.')
param modelName string = 'gpt-4.1-nano'

@description('Optional model version. Leave empty to let Azure choose the default supported version in the selected region.')
param modelVersion string = ''

@description('Deployment SKU name. GlobalStandard is commonly used for Azure OpenAI model deployments.')
param deploymentSkuName string = 'GlobalStandard'

@description('Deployment capacity. Start small and increase only after checking regional quota.')
@minValue(1)
param deploymentCapacity int = 10

@description('Controls whether an embedding model deployment should be created together with the account. This is skipped automatically when restore=true.')
param deployEmbeddingModel bool = true

@description('Embedding deployment name inside the Azure OpenAI account.')
param embeddingDeploymentName string = 'text-embedding-3-small'

@description('Azure OpenAI embedding model name to deploy.')
param embeddingModelName string = 'text-embedding-3-small'

@description('Optional embedding model version. Leave empty to let Azure choose the default supported version in the selected region.')
param embeddingModelVersion string = ''

@description('Embedding deployment SKU name. GlobalStandard is commonly used for Azure OpenAI model deployments.')
param embeddingDeploymentSkuName string = 'GlobalStandard'

@description('Embedding deployment capacity. Start small and increase only after checking regional quota.')
@minValue(1)
param embeddingDeploymentCapacity int = 10

@allowed([
  'Enabled'
  'Disabled'
])
@description('Whether the account should be reachable from the public internet.')
param publicNetworkAccess string = 'Enabled'

@description('Keep local auth enabled so API-key based clients can connect. Set to true only if you plan to use Entra-only auth.')
param disableLocalAuth bool = false

@description('Optional tags to apply to the account and deployment.')
param tags object = {}

@description('Controls whether the daily cleanup Logic App should be deployed together with the account.')
param deployDailyDeleteSchedule bool = false

@description('Hour of day to run the automatic delete job.')
@minValue(0)
@maxValue(23)
param deleteHour int = 20

@description('Minute of hour to run the automatic delete job.')
@minValue(0)
@maxValue(59)
param deleteMinute int = 0

@description('Time zone used by the automatic delete schedule.')
param deleteTimeZone string = 'UTC'

var accountProperties = union({
  customSubDomainName: accountName
  publicNetworkAccess: publicNetworkAccess
  disableLocalAuth: disableLocalAuth
}, restore ? {
  restore: true
} : {})

var cleanupLogicAppName = 'la-delete-${accountName}'

var deploymentModel = union({
  format: 'OpenAI'
  name: modelName
}, empty(modelVersion) ? {} : {
  version: modelVersion
})

var embeddingDeploymentModel = union({
  format: 'OpenAI'
  name: embeddingModelName
}, empty(embeddingModelVersion) ? {} : {
  version: embeddingModelVersion
})

// The account is always created first; the model deployment is optional so the
// same template can support account restore scenarios.
resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: accountName
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  tags: tags
  properties: accountProperties
}

resource modelDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployModel && !restore) {
  parent: account
  name: deploymentName
  sku: {
    name: deploymentSkuName
    capacity: deploymentCapacity
  }
  tags: tags
  properties: {
    model: deploymentModel
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
}

resource embeddingModelDeployment 'Microsoft.CognitiveServices/accounts/deployments@2024-10-01' = if (deployEmbeddingModel && !restore) {
  parent: account
  name: embeddingDeploymentName
  dependsOn: [
    modelDeployment
  ]
  sku: {
    name: embeddingDeploymentSkuName
    capacity: embeddingDeploymentCapacity
  }
  tags: tags
  properties: {
    model: embeddingDeploymentModel
    versionUpgradeOption: 'OnceNewDefaultVersionAvailable'
  }
}

// Deploy the cleanup workflow with the account so temporary environments can
// clean themselves up without a separate deployment step.
module dailyDeleteSchedule './azure-openai-daily-delete.bicep' = if (deployDailyDeleteSchedule) {
  name: 'daily-delete-schedule'
  dependsOn: [
    modelDeployment
    embeddingModelDeployment
  ]
  params: {
    location: location
    accountName: account.name
    embeddingDeploymentName: deployEmbeddingModel && !restore ? embeddingDeploymentName : ''
    logicAppName: cleanupLogicAppName
    deleteHour: deleteHour
    deleteMinute: deleteMinute
    timeZone: deleteTimeZone
    tags: tags
  }
}

output accountName string = account.name
output endpoint string = account.properties.endpoint
output deploymentName string = deployModel && !restore ? modelDeployment.name : ''
output embeddingDeploymentName string = deployEmbeddingModel && !restore ? embeddingModelDeployment.name : ''
output cleanupLogicAppName string = deployDailyDeleteSchedule ? cleanupLogicAppName : ''
output cleanupScheduledTime string = deployDailyDeleteSchedule ? '${padLeft(string(deleteHour), 2, '0')}:${padLeft(string(deleteMinute), 2, '0')} ${deleteTimeZone}' : ''
