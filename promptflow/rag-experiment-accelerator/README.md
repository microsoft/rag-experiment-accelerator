## Creating a custom environment for prompt flow runtime

```bash
az login

az configure --defaults workspace=$MLWorkSpaceName group=$ResourceGroupName

cd ./rag-experiment-accelerator

az ml environment create --file ./pf-context.yaml -w $MLWorkSpaceName
```