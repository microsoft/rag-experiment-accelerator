#!/bin/bash

pip install --upgrade pip

pip install -r requirements.txt

pip install -r dev-requirements.txt

pre-commit install
