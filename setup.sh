#!/bin/bash
set -euxo pipefail

# Upgrade pip
python3 -m pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt

# Install your package
pip install -e .

# Sanity check
python3 -c "import torch, pytest; print('✅ torch and pytest installed')"
