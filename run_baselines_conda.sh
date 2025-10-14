#!/bin/bash
# Script to create conda environment and run baselines

set -e  # Exit on error

echo "=========================================="
echo "Setting up conda environment..."
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Anaconda or Miniconda first."
    echo "Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Create conda environment if it doesn't exist
if conda env list | grep -q "^baseline_env "; then
    echo "Environment 'baseline_env' already exists. Activating..."
else
    echo "Creating new conda environment..."
    conda env create -f environment.yml
fi

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate baseline_env

# Verify installation
echo ""
echo "=========================================="
echo "Verifying installation..."
echo "=========================================="
python -c "import sklearn; print(f'✓ scikit-learn: {sklearn.__version__}')"
python -c "import xgboost; print(f'✓ xgboost: {xgboost.__version__}')"
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}')"
python -c "import tensorflow as tf; print(f'✓ TensorFlow: {tf.__version__}')" || echo "⚠ TensorFlow not available"
python -c "import pandas; print(f'✓ pandas: {pandas.__version__}')"
python -c "import numpy; print(f'✓ numpy: {numpy.__version__}')"

echo ""
echo "=========================================="
echo "Running baseline models (without DL)..."
echo "=========================================="

# Run baselines without deep learning (faster, more stable)
python ablation_study/run_generic_baselines.py \
  --data_path Dataset_and_train_sequence/antimicrobial_training_data.csv \
  --out_path ablation_study/baseline_results_conda.json \
  --seed 42 \
  --skip-dl

echo ""
echo "=========================================="
echo "✓ Training complete!"
echo "Results saved to: ablation_study/baseline_results_conda.json"
echo "=========================================="
