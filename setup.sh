from setuptools import setup, find_packages

setup(
    name='jepaforclaims',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch',
        'pytorch-lightning',
        'pandas',#!/bin/bash
set -euxo pipefail

# Upgrade pip
python3 -m pip install --upgrade pip

# Install dependencies
echo ">>> Installing requirements..."
pip install -r requirements.txt

# Install the editable package
echo ">>> Installing editable package..."
pip install -e .

# Sanity check
echo ">>> Verifying installed packages..."
python3 -c "import pytest; import torch; print('✅ Pytest and Torch successfully installed')"

        'scikit-learn',
        'numpy',
        'matplotlib',
        'seaborn',
        'tqdm',
    ],
)
