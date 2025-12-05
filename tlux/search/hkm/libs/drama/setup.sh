#!/bin/sh
python3 -m venv setup-env
source setup-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r setup_requirements.txt
python3 setup.py
deactivate
# rm -rf setup-env
