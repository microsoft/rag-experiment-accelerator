#!/bin/bash

pip install --upgrade pip

pip install -r requirements.txt

pip install -r dev-requirements.txt

python -m spacy download en_core_web_sm

# install the rag-accelerator packages in editable mode (required for pre-commit to work properly with pytest)
pip install -e .

pre-commit install
