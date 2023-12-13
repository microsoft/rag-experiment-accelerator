# Promptflow Secret Setup

## Prerequisites
Ensure that your `.env` file is properly created, instructions [here](../../../README.md#installation), install the dev-requirements and login to the az cli.
``` bash
# Install the dev requirements
pip install -r dev-requirements.txt 

# Login to the az cli
az login
```

## AzureML Connections
AzureML connections are recommended as the secrets are stored securely in Key Vault and config is stored in the workspace.

Update the `location` variable in `promptflow/rag-experiment-accelerator/setup/setup.py` to the location of your AzureML Workspace.

``` bash
# Set your promptflow connection provider to azureml
pf config set connection.provider=azureml

# Run the setup script
python promptflow/rag-experiment-accelerator/setup/setup.py 
```

## Local connections
You can create a custom local connection using the `.env`. This is not recommended.

``` bash
# Set your promptflow connection provider to local
pf config set connection.provider=local

# Create the prompt flow connection using your .env file
pf connection create -f .env --name rag_connection
```
