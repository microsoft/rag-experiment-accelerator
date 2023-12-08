## Creating a custom environment for prompt flow runtime

```bash
az configure --defaults workspace=$MLWorkSpaceName group=$ResourceGroupName

cd ./rag-experiment-accelerator

az ml environment create --file ./pf-context.yaml -w $MLWorkSpaceName
```