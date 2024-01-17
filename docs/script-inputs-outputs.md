# Script Overview

This document provides an overview of the scripts that are used to run the RAG Experiment Accelerator. The scripts are numbered in the order of execution, and each script has its own inputs and outputs.

## Prerequisites
Before running the scripts, you need to:

- Install the required packages and dependencies by following the instructions in the installation guide.
- Set up the required resources, such as Azure AI Search, Azure ML Studio, and Azure OpenAI by following the steps in the [required resources guide](/docs/environment-variables.md#required-resources).
- Create a `.env` file that contains the environment variables for the resources, such as subscription ID, resource group name, and service credentials. For more information on the `.env` file, see the [environment variables guide](/docs/environment-variables.md#environment-variables).

## 01_Index.py
This script creates and populates a search index in your Azure AI Search resource.

Inputs:

- A `/data` folder that contains the documents that you want to index. The `/data` folder (and its subfolders) can have documents in the following formats: PDF, HTML, Markdown, Text, JSON and Word (DOCX). For more information on how the documents are loaded, see the [documentLoader.py](/rag_experiment_accelerator/doc_loader/documentLoader.py) file.

Outputs:

- A `artifacts/generated_index_names.jsonl` file that contains the names of the generated search indexes.
- A populated search index in your Azure AI Search resource that contains the indexed documents and their metadata.

## 02_qa_generation.py
This script generates question-answer pairs from the indexed documents using the RAG model.

Inputs:

- The same `/data` folder as the previous step.

Outputs:

- A `artifacts/eval_data.jsonl` file that contains a list of JSON objects. Each JSON object has three fields: `user_prompt`, `output_prompt`, and `context`. The `user_prompt` field contains the generated question, the `output_prompt` field contains the generated answer, and the `context` field contains the document sections from which the question-answer pair was generated.

## 03_querying.py
This script queries the search index using the generated question-answer pairs and evaluates the search results.

Inputs:

- The `artifacts/eval_data.jsonl` file from the previous step.
- You can choose to provide your own `.jsonl` file (with same format as the one generated in the previous step) as input and update `config.json` with the full path of your file in the `eval_data_jsonl_file_path` field.
- `prompts_config.json` (optional)
  - You can provide a custom prompt to be used as the main prompt for the questions generated to search the data. The custom prompt can be provided as a string as follows:

```json
{"main_prompt_instruction": "<custom_main_prompt_instruction>"}
```



Outputs:

- A `artifacts/outputs/<CONFIG_VALUES>.jsonl` file that contains a list of JSON objects. Each JSON object has eleven fields: `actual`, `expected`, and nine fields relating to configuration information. The name of the output file is based on the values in the `config.json` file.
- The same output file is also uploaded to Azure Machine Learning Studio under `Assets -> Data`.

## 04_evaluation.py
This script calculates and displays the overall evaluation scores for the search index using the output files from the previous step.

Inputs:

- The `artifacts/outputs/<CONFIG_VALUES>.jsonl` file that was uploaded to Azure Machine Learning Studio.

Outputs:

- Several `artifacts/eval_score/*.csv` files that contain the calculated scores for different evaluation metrics, such as bert score, fuzzy score, and mean average precision. The name of the CSV file indicates the evaluation metric and the configuration values.
- The same CSV files are also found in Azure Machine Learning Studio at `Jobs -> <CONFIG_NAME> -> Outputs + Logs`.
