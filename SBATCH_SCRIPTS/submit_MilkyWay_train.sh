#!/bin/bash

# ============================================================================
# Master Submission Script for SIREN MilkyWay Pipeline
# ============================================================================
# This script submits a complete pipeline:
#   1. PREPARE (CPU) - Data preparation
#   2. TRAIN (GPU)   - Model training (depends on PREPARE)
#   3. TUNE (GPU)    - Hyperparameter tuning (depends on TRAIN)
#
# Usage:
#   ./submit_MilkyWay_train.sh
#
# Or with custom paths:
#   CONFIG_FILE=path/to/config.yaml \
#   REQUIREMENTS=path/to/requirements.txt \
#   VENV_PATH=path/to/venv \
# ============================================================================

echo "========================================================================"
echo "SIREN MilkyWay Pipeline Submission"
echo "========================================================================"
echo "Submission time: $(date)"
echo ""

# ============================================================================
# CONFIGURATION - Edit these paths as needed
# ============================================================================

# Path to config file (YAML)
export CONFIG_FILE="${CONFIG_FILE:-config.yaml}"

# Path to requirements file
export REQUIREMENTS="${REQUIREMENTS:-requirements.txt}"

# Path to virtual environment
export VENV_PATH="${VENV_PATH:-siren_env}"

# Individual submission scripts (should be in the same directory)
PREPARE_SCRIPT ="/idia/users/emilmeintjes/Road2SKA/ProjectF/SBATCH_SCRIPTS/slurm_PREPARE.sh"
TRAIN_SCRIPT   ="/idia/users/emilmeintjes/Road2SKA/ProjectF/SBATCH_SCRIPTS/slurm_TRAIN.sh"
TUNE_SCRIPT    ="/idia/users/emilmeintjes/Road2SKA/ProjectF/SBATCH_SCRIPTS/slurm_TUNE.sh"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

echo "Configuration:"
echo "  Config file:    $CONFIG_FILE"
echo "  Requirements:   $REQUIREMENTS"
echo "  Virtual env:    $VENV_PATH"
echo ""

# Create LOGS directory if it doesn't exist
if [ ! -d "LOGS" ]; then
    echo "Creating LOGS directory..."
    mkdir -p LOGS
fi

# Check if all submission scripts exist
MISSING_SCRIPTS=()
for script in "$PREPARE_SCRIPT" "$TRAIN_SCRIPT" "$TUNE_SCRIPT"; do
    if [ ! -f "$script" ]; then
        MISSING_SCRIPTS+=("$script")
    fi
done

if [ ${#MISSING_SCRIPTS[@]} -ne 0 ]; then
    echo "ERROR: The following submission scripts are missing:"
    for script in "${MISSING_SCRIPTS[@]}"; do
        echo "  - $script"
    done
    echo ""
    echo "Please ensure all scripts are in the current directory."
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS" ]; then
    echo "ERROR: Requirements file not found: $REQUIREMENTS"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found: $VENV_PATH"
    echo ""
    echo "Please create a virtual environment first:"
    echo "  python3 -m venv $VENV_PATH"
    exit 1
fi

echo "✓ All pre-flight checks passed"
echo ""

# ============================================================================
# JOB SUBMISSION WITH DEPENDENCIES
# ============================================================================

echo "------------------------------------------------------------------------"
echo "Submitting jobs with dependency chain..."
echo "------------------------------------------------------------------------"
echo ""

# Submit PREPARE job
echo "[1/3] Submitting PREPARE job..."
PREPARE_JOB=$(sbatch --parsable "$PREPARE_SCRIPT")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit PREPARE job"
    exit 1
fi

echo "      ✓ PREPARE job submitted: Job ID = $PREPARE_JOB"
echo ""

# Submit TRAIN job
echo "[2/3] Submitting TRAIN job (depends on PREPARE=$PREPARE_JOB)..."
TRAIN_JOB=$(sbatch --parsable --dependency=afterok:$PREPARE_JOB "$TRAIN_SCRIPT")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit TRAIN job"
    echo "      Cancelling PREPARE job $PREPARE_JOB..."
    scancel $PREPARE_JOB
    exit 1
fi

echo "      ✓ TRAIN job submitted: Job ID = $TRAIN_JOB"
echo ""

# Submit TUNE job
echo "[3/3] Submitting TUNE job (depends on TRAIN=$TRAIN_JOB)..."
TUNE_JOB=$(sbatch --parsable --dependency=afterok:$TRAIN_JOB "$TUNE_SCRIPT")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to submit TUNE job"
    echo "      Cancelling PREPARE and TRAIN jobs..."
    scancel $PREPARE_JOB $TRAIN_JOB
    exit 1
fi

echo "      ✓ TUNE job submitted: Job ID = $TUNE_JOB"
echo ""

# ============================================================================
# SUMMARY
# ============================================================================

echo "========================================================================"
echo "Pipeline submitted successfully!"
echo "========================================================================"
echo ""
echo "Job IDs:"
echo "  PREPARE: $PREPARE_JOB"
echo "  TRAIN:   $TRAIN_JOB (waits for PREPARE)"
echo "  TUNE:    $TUNE_JOB (waits for TRAIN)"
echo ""
echo "Dependency chain:"
echo "  $PREPARE_JOB → $TRAIN_JOB → $TUNE_JOB"
echo ""
echo "Log files will be written to:"
echo "  LOGS/slurm_PREPARE_${PREPARE_JOB}.out"
echo "  LOGS/slurm_TRAIN_${TRAIN_JOB}.out"
echo "  LOGS/slurm_TUNE_${TUNE_JOB}.out"
echo ""
echo "Monitor job status with:"
echo "  squeue -u \$USER"
echo ""
echo "Check specific job with:"
echo "  squeue -j $PREPARE_JOB,$TRAIN_JOB,$TUNE_JOB"
echo ""
echo "Cancel entire pipeline with:"
echo "  scancel $PREPARE_JOB $TRAIN_JOB $TUNE_JOB"
echo ""
echo "========================================================================"
echo "Submission completed at: $(date)"
echo "========================================================================"
