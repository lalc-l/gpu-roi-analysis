#!/bin/bash
# setup_environment.sh
# External Facing Price Performance General GPU benchmarking environment setup

set -e  # Exit on any error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Setting up GPU ROI Analysis Environment"
echo "======================================"
echo "Project root: $PROJECT_ROOT"
echo "Date: $(date)"
echo "User: $(whoami)"
echo "Node: $(hostname)"

# Detect GPU platform
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo "No GPU")
if echo "$GPU_INFO" | grep -q "H100"; then
    PLATFORM="h100"
    PLATFORM_UPPER="H100"
elif echo "$GPU_INFO" | grep -q "B200"; then
    PLATFORM="b200" 
    PLATFORM_UPPER="B200"
else
    echo "Warning: Could not detect H100 or B200 GPU"
    PLATFORM="unknown"
    PLATFORM_UPPER="UNKNOWN"
fi

echo "Platform detected: $PLATFORM_UPPER"
echo "GPU count: $(nvidia-smi --list-gpus 2>/dev/null | wc -l)"

# Create directory structure
echo "Creating directory structure..."
cd "$PROJECT_ROOT"
mkdir -p environments logs results/raw results/processed results/reports
mkdir -p scripts/benchmarks scripts/analysis scripts/utils
mkdir -p configs data

# Virtual environment setup
ENV_NAME="venv-${PLATFORM}"
ENV_PATH="environments/$ENV_NAME"

echo "Creating virtual environment: $ENV_PATH"
if [ -d "$ENV_PATH" ]; then
    echo "Removing existing environment..."
    rm -rf "$ENV_PATH"
fi

python3 -m venv "$ENV_PATH"
source "$ENV_PATH/bin/activate"

# Package installation
echo "Installing packages..."
pip install --upgrade pip --quiet

# Core ML packages
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121 --quiet
pip install transformers==4.36.0 --quiet
pip install accelerate==0.25.0 --quiet  
pip install datasets==2.15.0 --quiet

# System monitoring
pip install nvidia-ml-py3 --quiet
pip install psutil --quiet

# Data processing and visualization
pip install pandas==2.1.4 --quiet
pip install numpy==1.24.0 --quiet
pip install matplotlib==3.7.0 --quiet
pip install seaborn==0.12.2 --quiet

# Development tools
pip install jupyter --quiet
pip install tqdm==4.65.0 --quiet

# Optional performance packages
echo "Installing optional performance packages..."
pip install flash-attn==2.3.0 --no-build-isolation --quiet 2>/dev/null || echo "Flash Attention installation failed - continuing without it"

# Validation
echo "Validating installation..."
python -c "
import sys
import torch
import transformers
import pandas as pd
import numpy as np

print('Python version:', sys.version.split()[0])
print('PyTorch version:', torch.__version__)
print('Transformers version:', transformers.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU name:', torch.cuda.get_device_name(0))
    print('GPU memory (GB):', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1))
    print('GPU count:', torch.cuda.device_count())
print('Pandas version:', pd.__version__)
print('NumPy version:', np.__version__)
"

# Create activation script
ACTIVATE_SCRIPT="environments/activate_${PLATFORM}.sh"
cat > "$ACTIVATE_SCRIPT" << EOF
#!/bin/bash
# Activation script for $PLATFORM_UPPER environment

cd ~/benchmarks/gpu-roi-analysis
source environments/$ENV_NAME/bin/activate

echo "ROI Analysis Environment Active"
echo "Platform: $PLATFORM_UPPER"
echo "Python: \$(which python)"
echo "Working directory: \$(pwd)"

# Export environment variables
export PLATFORM="$PLATFORM_UPPER"
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-"all"}
export TOKENIZERS_PARALLELISM=false
EOF

chmod +x "$ACTIVATE_SCRIPT"

# Create requirements.txt
pip freeze > requirements.txt

# Log setup completion  
LOG_DIR="logs"
SETUP_LOG="$LOG_DIR/setup_${PLATFORM}_$(date +%Y%m%d_%H%M%S).log"
cat > "$SETUP_LOG" << EOF
Setup completed: $(date)
Platform: $PLATFORM_UPPER  
Node: $(hostname)
User: $(whoami)
Environment: $ENV_PATH
Python: $(which python)
GPU Info: $GPU_INFO
Package count: $(pip list | wc -l)
EOF

echo ""
echo "Setup completed successfully"
echo "Log saved to: $SETUP_LOG"
echo ""
echo "To activate environment:"
echo "  source $ACTIVATE_SCRIPT"
echo ""
echo "To run benchmarks:"
echo "  source $ACTIVATE_SCRIPT"
echo "  python scripts/benchmarks/run_benchmark.py --platform $PLATFORM"
