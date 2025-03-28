#!/bin/bash

START_COMMAND="python api-server.py"


# Create and activate Conda environment
echo "Setting up Conda environment..."
conda create -y -n "zonos" python="3.10"
source activate "zonos" || conda activate "zonos"

# Install dependencies
echo "Installing dependencies..."
pip install -e .


# Start the model server
echo "Starting Zonos server..."
eval "$START_COMMAND"
