#!/bin/bash
#SBATCH --job-name=PREP
#SBATCH --output=LOGS/prepare_%j.out
#SBATCH --error=LOGS/prepare_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=Main
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB

# SIREN Data Preparation - CPU
# This script prepares the FITS data for training

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

# ============================================================================
# CONFIGURATION - Edit these paths as needed
# ============================================================================
CONFIG_FILE="${CONFIG_FILE:-config.yaml}"
REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"
VENV_PATH="${VENV_PATH:-siren_env}"

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

# Load modules
module purge
module load python/3.9

# Activate virtual environment
echo ""
echo "Activating virtual environment: $VENV_PATH"
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi
source $VENV_PATH/bin/activate

# Install requirements
echo ""
echo "Installing requirements from: $REQUIREMENTS"
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: Requirements file not found at $REQUIREMENTS"
    exit 1
fi
pip install -r $REQUIREMENTS --quiet

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

echo ""
echo "=========================================="
echo "Validation Checks"
echo "=========================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi
echo "✓ Config file found: $CONFIG_FILE"

# Check if prepare.py exists
if [ ! -f "prepare.py" ]; then
    echo "ERROR: prepare.py not found in current directory"
    exit 1
fi
echo "✓ prepare.py found"

# Check Python installation
python --version || { echo "ERROR: Python not available"; exit 1; }
echo "✓ Python available"

echo "=========================================="
echo ""

# ============================================================================
# RUN DATA PREPARATION
# ============================================================================

echo "Starting data preparation..."
echo ""

python prepare.py --config "$CONFIG_FILE"

PREPARE_EXIT_CODE=$?

if [ $PREPARE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "ERROR: Data preparation failed with exit code $PREPARE_EXIT_CODE"
    exit $PREPARE_EXIT_CODE
fi

echo ""
echo "=========================================="
echo "Data preparation completed successfully"
echo "End time: $(date)"
echo "=========================================="

# Deactivate virtual environment
deactivate
