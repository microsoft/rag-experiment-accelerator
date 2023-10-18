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
git clone https://github.com/microsoft/rag-experiment-accelerator.git
```

2. **setup env file**: create .env file at top folder level and provide data for items mentioned:

```bash



AZURE_SEARCH_SERVICE_ENDPOINT=
AZURE_SEARCH_ADMIN_KEY=
OPENAI_ENDPOINT=
OPENAI_API_KEY=
OPENAI_API_VERSION=
OPENAI_EMBEDDING_DEPLOYED_MODEL=
SUBSCRIPTION_ID=
WORKSPACE_NAME=
RESOURCE_GROUP_NAME=

```
3. Execute the requirements.txt in a conda (first install Anaconda/Miniconda) or virtual environment (then install a couple of dependencies - promted on the run) to install the dependencies.

```bash
conda create -n rag-test python=3.10
conda activate rag-test
python -m pip install -r requirements.txt
```

4. Install Azure CLI and authorize:
```bash
az login
az account set  --subscription="<your_subscription_guid>"
az account show
```

5. Copy your `.pdf` files into the `data` folder.

## How to use

To harness the capabilities of the **RAG Experiment Accelerator**, follow these steps:

1. Modify the `search_config.json` file with hyper-parameters relevant to the experiment.
2. Execute `01_index.py` (python 01_index.py) to generate Azure Cognitive Search indexes and ingest data into the indexes.
3. Execute `02_qa_generation.py` (python 02_qa_generation.py) to generate Question-Answer pairs using Azure OpenAI.
4. Execute `03_querying.py` (python 03_querying.py) to query Azure Cognitive Search to generate context, re-ranking items in context and get response from Azure OpenAI using the new context. 
5. Execute `04_evaluation.py` (python 04_evaluation.py)  to calculate metrics using multiple metrics and generate incremental charts and reports in AzureML using MLFLOW integration.


# Description of configuration elements

```json
{
    "name_prefix": "Name of experiment, search index name used for tracking and comparing jobs",
    "chunking": {
        "chunk_size": "Size of each chunk e.g. [500, 1000, 2000]" ,
        "overlap_size": "Overlap Size for each chunk e.g. [100, 200, 300]" 
    },
    "embedding_dimension" : "embedding Size for each chunk e.g. [384, 1024]. Valid values are 384, 768,1024" ,
    "efConstruction" : "efConstruction value determines the value of Azure Cognitive Search vector configuration." ,
    "efsearch":  "efsearch value determines the value of Azure Cognitive Search vector configuration.",
    "rerank": "determines if search results should be re-ranked. Value values are TRUE or FALSE" ,
    "rerank_type": "determines the type of re-ranking. Value values are llm or crossencoder", 
    "llm_re_rank_threshold": "determines the threshold when using llm re-ranking. Chunks with rank above this number are selected in range from 1 - 10." ,
    "cross_encoder_at_k": "determines the threshold when using cross-encoding re-ranking. Chunks with given rank value are selected." ,
    "crossencoder_model" :"determines the model used for cross-encoding re-ranking step. Valid value is cross-encoder/stsb-roberta-base",
    "search_types" : "determines the search types used for experimentation. Valid value are search_for_match_semantic, search_for_match_Hybrid_multi, search_for_match_Hybrid_cross, search_for_match_text, search_for_match_pure_vector, search_for_match_pure_vector_multi, search_for_match_pure_vector_cross, search_for_manual_hybrid. e.g. ['search_for_manual_hybrid', 'search_for_match_Hybrid_multi','search_for_match_semantic' ]",
    "retrieve_num_of_documents": "determines the number of chunks to retrieve from the search index",
    "metric_types" : "determines the metrics used for evaluation purpose. Valid value are lcsstr, lcsseq, cosine, jaro_winkler, hamming, jaccard, levenshtein, fuzzy, bert_all_MiniLM_L6_v2, bert_base_nli_mean_tokens, bert_large_nli_mean_tokens, bert_large_nli_stsb_mean_tokens, bert_distilbert_base_nli_stsb_mean_tokens, bert_paraphrase_multilingual_MiniLM_L12_v2. e.g ['fuzzy','bert_all_MiniLM_L6_v2','cosine','bert_distilbert_base_nli_stsb_mean_tokens']",
    "chat_model_name":  "determines the OpenAI model" ,
    "openai_temperature": "determines the OpenAI temperature. Valid value ranges from 0 to 1.",
    "search_relevancy_threshold": "the similarity threshold to determine if a doc is relevant. Valid ranges are from 0.0 to 1.0"
}
```

## Reports

The solution is integrated with AzureML and uses MLFlow to manage experiments, jobs and artifacts. Following are some of the reports generated as part of evaluation flow.

### Metric Comparison

![Metric Comparison](./images/metric_comparison.png)

### Metric Analysis

![Alt text](./images/metric_analysis.png)

### Hyper Parameters

![Hyper Parameters](./images/hyper-parameters.png)

### Sample Metrics

![Sample Metrics](./images/sample_metric.png)

### Search evaluation

![Search evaluation](./images/search_chart.png)


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
