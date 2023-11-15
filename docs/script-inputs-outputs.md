# Script Overview

## Prerequisites

Before running the scripts:

- Follow installation instructions [here](/README.md#installation)
  - For more information on required resources, see [required resources](/docs/environment-variables.md#required-resources)
  - For more information on the .env file, see [.env file](/docs/environment-variables.md#environment-variables)

## 01_index.py

Inputs:

- Data must exist in the `/data` folder

  - `/data` folder can have 4 potential subfolders for: pdf, html, markdown, text, see [documentLoader.py](/rag_experiment_accelerator/doc_loader/documentLoader.py)

Outputs:

- `artifacts/generated_index_names.jsonl`
- populated search index in your Azure Cognitive Search resource

## 02_qa_generation.py

Inputs:

- Sample `/data` folder as the previous step

Outputs:

- `artifacts/eval_data.jsonl`
  - list of json that contain `user_prompt`, `output_prompt`, and `context`

## 03_querying.py

Inputs:

- `artifacts/eval_data.jsonl` (default)

  - You can choose to provide your own `.jsonl` file as input and update the path of that file in the `search_config.json` file in the `eval_data_jsonl_file_path` field.

- `prompts_config.json` (optional)

  - You can use custom prompt(s) to be used as main prompt(s) for the questions generated to search the data. The custom prompts can be provided as a list of strings as follows:

```json
{"main_prompt_instructions": ["<main_prompt_1>", "<main_prompt_2>"]}
```

Outputs:

- `artifacts/outputs/<CONFIG_VALUES>.jsonl`

  - name of your output json is based on the values in your search_config.json
  - list of json that contain config information, query information, and search_eval scores
  - `<CONFIG_VALUES>.jsonl` also gets uploaded to Azure ML Studio under `Assets -> Data`

## 04_evaluation.py

Inputs:

- `artifacts/outputs/<CONFIG_VALUES>.jsonl` uploaded to Azure ML Studio

Outputs:

- `artifacts/eval_score/*.csv`
  - various csv's with calculated scores
  - also found in Azure ML Studio at `Jobs -> <CONFIG_NAME> -> Outputs + Logs`
