#!/bin/bash

pip install --upgrade pip

pip install -r requirements.txt

pip install -r dev-requirements.txt

# install the rag-accelerator package in editable mode (requiered for pre-commit to work properly with pytest)
pip install -e .

pre-commit install
