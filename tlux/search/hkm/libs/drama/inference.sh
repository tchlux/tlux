#!/bin/sh
python3 -m venv inference-env
source inference-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r inference_requirements.txt
python3 inference.py
deactivate
