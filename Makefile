include .env
export

MAIN_BICEP_FILE = "./infra/main.bicep"

deploy:
	az deployment group create \
	--resource-group $(RESOURCE_GROUP_NAME) \
	--template-file "$(MAIN_BICEP_FILE)"

deploy_with_isolated_network:
	az deployment group create \
		--resource-group $(RESOURCE_GROUP_NAME) \
		--template-file "$(MAIN_BICEP_FILE)" \
		--parameters DeployResourcesWithIsolatedNetwork=true \
		--parameters VnetAddressSpace=$(VIRTUAL_NETWORK_ADDRESS_SPACE) \
		--parameters ProxySubnetAddressSpace=$(PROXY_SERVER_SUBNET_ADDRESS_SPACE) \
		--parameters AzureSubnetAddressSpace=$(AZURE_RESOURCES_SUBNET_ADDRESS_SPACE)

hello_world:
	@echo "Hello, World!"
