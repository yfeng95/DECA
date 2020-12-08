#!/usr/bin/env bash

export CONDA_ENV_NAME=deca-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
pip install -r requirements.txt