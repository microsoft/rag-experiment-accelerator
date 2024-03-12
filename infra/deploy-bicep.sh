# Set your resource group name and location
resourceGroupName="aj-test-rag-deployment"
location="uksouth"

# Deploy the Bicep template
az deployment group create \
  --name rag-deployment1 \
  --resource-group $resourceGroupName \
  --template-file "test_network_isolation.bicep"
