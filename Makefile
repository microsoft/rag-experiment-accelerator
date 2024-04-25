.PHONY: all index qnagen query eval help azureml clear_docs clear_artifacts test flake updatekv
.DEFAULT_GOAL := help

# Load .env file if exists and export all variables before running any target
ENV_FILE := .env
ifeq ($(filter $(MAKECMDGOALS),config clean),)
	ifneq ($(strip $(wildcard $(ENV_FILE))),)
		ifneq ($(MAKECMDGOALS),config)
			include $(ENV_FILE)
			export
		endif
	endif
endif

SHELL := /bin/bash
target_title = @echo -e "\n\e[34m»»» 🧩 \e[96m$(1)\e[0m..."

help: ## 💬 This help message :)
	@grep -E '[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

all: index qnagen query eval ## 🔠 Run all steps in sequence: load_env index qnagen query eval
query_eval: query eval ## 🔍👓 Run query and eval steps in sequence: load_env query eval

load_env: ## 📃 Load .env file
	$(call target_title, "loading env file") \
	&& source .env

index: ## 📚 Index documents (download documents from blob storage, split to chunks, generate embeddings, create and upload to azure search index)
	$(call target_title, "indexing")
	python3 01_index.py $(if $(dd),--data_dir $(dd), --data_dir ./data) $(if $(cp),--config_path $(cp))

qnagen: ## ❓ Generate questions and answers for all document chunks in configured index
	$(call target_title, "question and answer generation")
	python3 02_qa_generation.py $(if $(dd),--data_dir $(dd), --data_dir ./data) $(if $(cp),--config_path $(cp))

query: ## 🔍 Query the index for all questions in jsonl file configured in config.json and generate answers using LLM
	$(call target_title, "querying")
	python3 03_querying.py $(if $(dd),--data_dir $(dd), --data_dir ./data) $(if $(cp),--config_path $(cp))

eval: ## 👓 Evaluate metrics for all answers compared to ground truth
	$(call target_title, "evaluating")
	python3 04_evaluation.py $(if $(dd),--data_dir $(dd), --data_dir ./data) $(if $(cp),--config_path $(cp))


azureml: ## 🚀 Run all steps in sequence on Azure ML
	$(call target_title, "running on Azure ML")
	python3 azureml/pipeline.py $(if $(dd),--data_dir $(dd), --data_dir ./data)  $(if $(cp),--config_path $(cp), --config_path ./config.json)


clear_docs: ## ❌ Delete all downloaded documents from data folder
	$(call target_title, "deleting all downloaded documents from data folder")
	rm -rf data

clear_artifacts: ## ❌ Delete all document chunks, index data and evaluation scores from artifacts folder
	$(call target_title, "clearing artifacts folder") \
	&& rm -rf ./artifacts/docs_data \
	&& rm -rf ./artifacts/eval_score \
	&& rm -rf ./artifacts/index_data \
	&& rm -rf ./artifacts/outputs

test: ## 🧪 Run tests
	$(call target_title, "running tests")
	pytest . --cov=. --cov-report=html --cov-config=.coveragerc

flake: ## 🧹 Run flake8
	$(call target_title, "running flake8")
	flake8 --extend-ignore=E501

updatekv: ## 🔄 Update keyvault secrets
	$(call target_title, "updating keyvault secrets")
	python3 env_to_keyvault.py
