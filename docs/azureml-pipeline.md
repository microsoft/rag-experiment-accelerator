# AzureML Pipeline

## What is it

RAG Experiment Accelerator supports running the pipeline on Azure ML compute, in addition to running it on local machine. This document describes the steps to deploy the necessary resources on Azure, and to run the pipeline on Azure ML compute. 

## Architecture diagram

![Architecture diagram](../images/azureml_pipeline.png)

![AzureML pipeline overview](../images/azureml_pipeline_overview.png)

## Deploying the cloud resources 

To run the steps of RAG experiment accelerator on Azure ML compute, you will need perform the steps from [installation](../README.md#installation) section of the README file. This will take you through the setup to deploy all necessary Azure configuration for you.

After this, follow these steps to deploy additional resources to be able to run Azure ML pipeline:

1. Deploy an AML Compute cluster. In your Azure ML workspace, navigate to Compute -> Compute clusters, and click New. Pick the virtual machine required, give it a name, and set minimum and maximum number of nodes. It is recommended to set the minimum number of nodes to 0, so that the cluster can scale down to 0. In Advanced settings, assign a system-assigned managed identity.

![Screenshot detailing the steps to create AML Compute cluster](../images/create_compute_cluster.png)

2. When you created an Azure ML workspace, Azure will have created an associated keyvault for you. In the keyvault, create two Access Policies: one for yourself (or for the principal who is going to run the pipeline), and another one for the system-assigned managed identity created in the previous step. Both of these principals need Get, Set and List permissions on secrets.

![Screenshot detailing the steps to create access policies](../images/create_access_policies.png)

3. Add the following environment variables to the .env file:

```bash
AML_COMPUTE_NAME=<name of the compute you have created>
AML_COMPUTE_INSTANCES_NUMBER=<maximum number of instances in the cluster>
AZURE_KEY_VAULT_ENDPOINT=<keyvault endpoint, e.g. https://<keyvault-name>.vault.azure.net>
```

4. Populate the keyvault with the environment configuration. To do so, run the script

```bash
python env_to_keyvault.py
```

This will create a secret in the keyvault for each environment variable in the .env file.

## Running the pipeline

To run the pipeline, run the following command:

```bash
python azureml/pipeline.py
```

This will submit the pipeline to Azure ML compute, and you will be able to monitor the run in the Azure ML workspace.

On AzureML, head to your workspace and select Jobs on the left. Find the name for your experiment as defined by the name in the config. You will see all runs for your experiment. Click on the latest run to see the details of the run.

![Screenshot detailing the steps to view the run](../images/view_list_of_runs.png)

You also can visually compare the metrics for each run by using Dashboard tab. 

## Troubleshooting the pipeline

If the pipeline fails, you can check the logs of the run by clicking on the failed run, and then clicking on the logs tab. This will show you the logs for each step of the pipeline under `Outputs + logs` tab.

![Screenshot detailing the steps to view logs](../images/view_logs.png)

If the first step (the indexing step) of the pipeline fails, you may not be able to see the logs under that tab. You may need to look at the error logs for each individual compute instance or even each individual node.

![Screenshot detailing the steps to view logs](../images/view_logs_parallel_step.png)