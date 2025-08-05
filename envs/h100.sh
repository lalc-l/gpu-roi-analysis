#!/bin/bash
cd ~/ext-bench-fs/benchmarks/gpu-roi-analysis
source envs/venv-h100/bin/activate
echo "ROI Analysis Environment Active"
echo "Platform: H100"
echo "Python: $(which python)"
echo "Working directory: $(pwd)"
export PLATFORM="H100"
export TOKENIZERS_PARALLELISM=false
