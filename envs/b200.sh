#!/bin/bash
# Activation script for B200 environment

# Navigate to project directory
cd ~/benchmarks/gpu-roi-analysis

# Activate virtual environment
source envs/venv-b200/bin/activate

echo "=== ROI Analysis Environment Active ==="
echo "Platform: B200"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"

# Export environment variables
export PLATFORM="B200"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"all"}
export TOKENIZERS_PARALLELISM=false
