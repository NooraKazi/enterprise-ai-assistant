targetScope = 'resourceGroup'

// This template schedules a daily deletion of the Azure OpenAI account.
// It intentionally deletes only the target account resource, not the entire
// resource group. If you need full resource-group deletion, the scheduler must
// live in a different resource group or at subscription scope.

@description('Location for the cleanup workflow.')
param location string = resourceGroup().location

@description('Name of the existing Azure OpenAI account to delete on schedule.')
param accountName string = 'ai-enterprise-ai-assistant'

@description('Optional name of an existing embedding deployment to delete before the account is deleted.')
param embeddingDeploymentName string = ''

@description('Name of the Logic App that runs the daily cleanup job.')
param logicAppName string = 'la-delete-openai-daily'

@description('Hour of day to run the deletion job.')
@minValue(0)
@maxValue(23)
param deleteHour int = 20

@description('Minute of hour to run the deletion job.')
@minValue(0)
@maxValue(59)
param deleteMinute int = 0

@description('Time zone used by the recurrence trigger, for example UTC or Arab Standard Time.')
param timeZone string = 'UTC'

@description('Optional tags applied to the Logic App.')
param tags object = {}

var workflowSchema = 'https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#'
var contributorRoleDefinitionId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', 'b24988ac-6180-42a0-ab88-20f7382dd24c')
var accountResourceId = resourceId('Microsoft.CognitiveServices/accounts', accountName)
var embeddingDeploymentResourceId = empty(embeddingDeploymentName) ? '' : resourceId('Microsoft.CognitiveServices/accounts/deployments', accountName, embeddingDeploymentName)
var accountApiVersion = '2024-10-01'
var workflowActions = union(empty(embeddingDeploymentName) ? {} : {
  Delete_embedding_model: {
    type: 'Http'
    inputs: {
      method: 'DELETE'
      uri: '${environment().resourceManager}${embeddingDeploymentResourceId}?api-version=${accountApiVersion}'
      authentication: {
        type: 'ManagedServiceIdentity'
        audience: environment().resourceManager
      }
    }
    runAfter: {}
  }
}, {
  Delete_openai_account: {
    type: 'Http'
    inputs: {
      method: 'DELETE'
      uri: '${environment().resourceManager}${accountResourceId}?api-version=${accountApiVersion}'
      authentication: {
        type: 'ManagedServiceIdentity'
        audience: environment().resourceManager
      }
    }
    runAfter: empty(embeddingDeploymentName) ? {} : {
      Delete_embedding_model: [
        'Succeeded'
        'Failed'
      ]
    }
  }
})

resource account 'Microsoft.CognitiveServices/accounts@2024-10-01' existing = {
  name: accountName
}

// The Logic App uses a managed identity plus a scoped role assignment so the
// scheduled cleanup can delete only the target Azure OpenAI account.
resource cleanupWorkflow 'Microsoft.Logic/workflows@2019-05-01' = {
  name: logicAppName
  location: location
  identity: {
    type: 'SystemAssigned'
  }
  tags: union(tags, {
    purpose: 'daily-openai-cleanup'
  })
  properties: {
    state: 'Enabled'
    definition: {
      '$schema': workflowSchema
      contentVersion: '1.0.0.0'
      triggers: {
        Recurrence: {
          type: 'Recurrence'
          recurrence: {
            frequency: 'Day'
            interval: 1
            schedule: {
              hours: [
                deleteHour
              ]
              minutes: [
                deleteMinute
              ]
            }
            timeZone: timeZone
          }
        }
      }
      actions: workflowActions
      outputs: {}
    }
  }
}

resource cleanupRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: account
  name: guid(account.id, cleanupWorkflow.name, 'daily-delete-contributor')
  properties: {
    roleDefinitionId: contributorRoleDefinitionId
    principalId: cleanupWorkflow.identity.principalId
    principalType: 'ServicePrincipal'
    description: 'Allows the cleanup Logic App to delete the Azure OpenAI account.'
  }
}

output cleanupLogicAppName string = cleanupWorkflow.name
output cleanupLogicAppIdentityPrincipalId string = cleanupWorkflow.identity.principalId
output targetAccountResourceId string = account.id
output targetEmbeddingDeploymentResourceId string = embeddingDeploymentResourceId
output scheduledTime string = '${padLeft(string(deleteHour), 2, '0')}:${padLeft(string(deleteMinute), 2, '0')} ${timeZone}'
