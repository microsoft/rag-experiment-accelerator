#!/bin/bash

source ../.env

# Deploy the Bicep template
az deployment group create \
  --name $VNET_DEPLOYMENT_NAME \
  --vnet-address-space $VIRTUAL_NETWORK_ADDRESS_SPACE \
  --proxy-server-subnet-address-prefix $PROXY_SERVER_SUBNET_ADDRESS_SPACE \
  --az-resources-subnet-address-space $AZURE_RESOURCES_SUBNET_ADDRESS_SPACE
  --resource-group $RESOURCE_GROUP_NAME \
  --template-file "main.bicep"
