# RAG Experiment Accelerator

## Overview

The **RAG Experiment Accelerator** is a versatile tool designed to expedite and facilitate the process of conducting experiments and evaluations using Azure Cognitive Search and RAG pattern. This document offers an extensive guide detailing everything you need to know about this tool, including its purpose, features, installation, usage, and more.

## Purpose

The primary objective of the **RAG Experiment Accelerator** is to streamline and simplify the intricate task of running experiments and evaluations of search queries and quality of response from OpenAI. This tool is valuable for researchers, data scientists, and developers seeking to:

- Assess the performance of diverse Search and OpenAI related hyper-parameters.
- Compare the efficacy of various search strategies.
- Fine-tune and optimize parameters.
- Find the best combination of hyper-parameters.
- Generate comprehensive reports and visualizations from experiment results.

## Features

The **RAG Experiment Accelerator** is config driven and offers a robust array of features to support its purpose:

1. **Experiment Setup**: Users can meticulously define and configure experiments by specifying an array of search engine parameters, search types, query sets, and evaluation metrics.

2. **Integration**: Seamlessly integrates with Azure Cognitive Search, Azure Machine Learning, MLFlow and Azure OpenAI.

3. **Rich Search Index**: generates multiple search indexes based on hyper-parameter configurations available in config file.

4. **Query Generation**: The tool is equipped with the capability to generate an assortment of diverse, customizable query sets, which can be tailor-made for specific experimentation requirements.

5. **multiple search types**: It supports multiple search types including pure text, pure vector, cross-vector, multi-vector, hybrid plus more empowering users with the capability to conduct comprehensive analysis on seach capabilities and eventual results.

6. **sub-quering**: The pattern evaluates the user query and if it find it complex enough, it would break it down into smalle sub-queries to generate relevant context.

7. **re-ranking**: The query responses from Azure Cognitive Search are re-evaluated using LLM and ranked them according to relevancy between the query and the context.

8. **Metrics and Evaluation**: Users are afforded the liberty to define custom evaluation metrics, thereby enabling precise and granular assessment of search algorithm performance. It includes distance based, cosine, semantic similarity plus more metrics out of the box.

9. **Report Generation**: The **RAG Experiment Accelerator** automates the process of report generation, complete with visually compelling visualizations that facilitate effortless analysis and effortless sharing of experiment findings.


## Installation

To harness the capabilities of the **RAG Experiment Accelerator**, follow these installation steps:

1. **Clone the Repository**: Begin by cloning the accelerator's repository from its [GitHub repository](https://github.com/your-repo/search-experiment-accelerator.git).

```bash
git clone https://github.com/your-repo/rag-experiment-accelerator.git
```

## How to use

To harness the capabilities of the **RAG Experiment Accelerator**, follow these steps:

1. Modify the search_config.json file with hyper-parameters relevant to the experiment.
2. Execute 01_index.py to generate Azure Cognitive Search indexes and ingest data into the indexes.
3. Execute 02_qa_generation.py to generate Question-Answer pairs using Azure OpenAI.
4. Execute 03_querying.py to query Azure Cognitive Search to generate context, re-ranking items in context and get response from Azure OpenAI using the new context.
5. Execute 04_evaluation.py to calculate metrics using multiple metrics and generate incremental charts and reports in AzureML using MLFLOW integration.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
