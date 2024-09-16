#!/bin/bash

# Install system dependencies for unstructured
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx poppler-utils tesseract-ocr

# Install Python dependencies
pip install --upgrade pip

pip install -r requirements.txt

pip install -r dev-requirements.txt

python -m spacy download en_core_web_sm

# install the rag-accelerator packages in editable mode (required for pre-commit to work properly with pytest)
pip install -e .

pre-commit install
