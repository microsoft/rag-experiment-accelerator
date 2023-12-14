## Creating a custom environment for prompt flow runtime

```bash
az login

az account set --subscription <subscription ID>

az extension add --name ml

az configure --defaults workspace=$MLWorkSpaceName group=$ResourceGroupName

cd ./promptflow/rag-experiment-accelerator/custom_environment

az ml environment create --file ./environment.yaml -w $MLWorkSpaceName
```