# RAG Experiment Accelerator

## Overview

The **RAG Experiment Accelerator** is a versatile tool that helps you conduct experiments and evaluations using Azure Cognitive Search and RAG pattern. This document provides a comprehensive guide that covers everything you need to know about this tool, such as its purpose, features, installation, usage, and more.

## Purpose

The main goal of the **RAG Experiment Accelerator** is to make it easier and faster to run experiments and evaluations of search queries and quality of response from OpenAI. This tool is useful for researchers, data scientists, and developers who want to:

- Test the performance of different Search and OpenAI related hyperparameters.
- Compare the effectiveness of various search strategies.
- Fine-tune and optimize parameters.
- Find the best combination of hyperparameters.
- Generate detailed reports and visualizations from experiment results.

## Features

The **RAG Experiment Accelerator** is config driven and offers a rich set of features to support its purpose:

1. **Experiment Setup**: You can define and configure experiments by specifying a range of search engine parameters, search types, query sets, and evaluation metrics.

2. **Integration**: It integrates seamlessly with Azure Cognitive Search, Azure Machine Learning, MLFlow and Azure OpenAI.

3. **Rich Search Index**: It creates multiple search indexes based on hyperparameter configurations available in the config file.

4. **Query Generation**: The tool can generate a variety of diverse and customizable query sets, which can be tailored for specific experimentation needs.

5. **Multiple Search Types**: It supports multiple search types, including pure text, pure vector, cross-vector, multi-vector, hybrid, and more. This gives you the ability to conduct comprehensive analysis on search capabilities and results.

6. **Sub-Querying**: The pattern evaluates the user query and if it finds it complex enough, it breaks it down into smaller sub-queries to generate relevant context.

7. **Re-Ranking**: The query responses from Azure Cognitive Search are re-evaluated using LLM and ranked according to the relevance between the query and the context.

8. **Metrics and Evaluation**: You can define custom evaluation metrics, which enable precise and granular assessment of search algorithm performance. It includes distance-based, cosine, semantic similarity, and more metrics out of the box.

9. **Report Generation**: The **RAG Experiment Accelerator** automates the process of report generation, complete with visually appealing visualizations that make it easy to analyze and share experiment findings.

10. **Multi-Lingual**: The tool supports language analyzers for linguistic support on individual languages and specialized (language-agnostic) analyzers for user-defined patterns on search indexes. For more information, see [Types of Analyzers](https://learn.microsoft.com/en-us/azure/search/search-analyzers#types-of-analyzers).


## How to use it

See Samples folder with samples of how to use it.

## Contributing

We welcome your contributions and suggestions. To contribute, you need to agree to a
Contributor License Agreement (CLA) that confirms you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [https://cla.opensource.microsoft.com].

When you submit a pull request, a CLA bot will automatically check whether you need to provide
a CLA and give you instructions (for example, status check, comment). Follow the instructions
from the bot. You only need to do this once for all repos that use our CLA.

Before you contribute, make sure to run

```
pip install -e .
pre-commit install
```

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions or comments.

### Developer Contribution Guidelines

- **Branch Naming Convention**: 
    - Use the GitHub UI to include a tag in the branch name, then create the branch directly from the UI. Here are some examples:
        - `bug/11-short-description`
        - `feature/22-short-description`
- **Merging Changes**: 
    - When merging, squash your commits to have up to 3 incremental commits for Pull Requests (PRs) and merges.
- **Branch Hygiene**: 
    - Delete the branch after it has been merged.
- **Testing Changes Locally**: 
    - Before merging, test your changes locally.
- **Naming Conventions**: 
    - Use snake case for metric names and configuration variables, like `example_snake_case`.
    - Set up your Git username to be your first and last name, like this: `git config --global user.name "First Last"`
- **Issue Tracking**:
    - Working on a contribution to the RAG Experiment Accelerator? Before opening a new issue, make sure to check if the feature has already been requested by searching for it in the associated [project issue tracker](https://github.com/orgs/microsoft/projects/991), and consider adding to that discussion instead. Otherwise, please open an issue for it using the feature request template or create a PR and make sure it is associated to the [project](https://github.com/orgs/microsoft/projects/991).


## Trademarks

This project might contain trademarks or logos for projects, products, or services. You must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general) to use Microsoft 
trademarks or logos correctly.
Don't use Microsoft trademarks or logos in modified versions of this project in a way that causes confusion or implies Microsoft sponsorship.
Follow the policies of any third-party trademarks or logos that this project contains.
