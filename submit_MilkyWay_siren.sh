#!/bin/bash
#SBATCH --job-name=PIXELS
#SBATCH --output=LOGS/Milky_Way_siren_%j.out
#SBATCH --error=LOGS/Milky_Way_siren_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=GPU
#SBATCH --reservation=roadtoska-gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

echo "=========================================="
echo "SLURM Job Information"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "=========================================="

module purge
module load singularity

# P100-compatible container
CONTAINER=/idia/software/containers/ASTRO-GPU-PyTorch-2023-10-10.sif

# Output directory
export OUTPUT_DIR="$PWD/results_${SLURM_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

# FITS file
FITS_FILE="./MeerKAT_Galactic_Centre_1284MHz-StokesI.fits"

# Training parameters
STEPS=10000
HIDDEN_FEATURES=256
HIDDEN_LAYERS=3
BATCH_SIZE=2048
SIDELENGTH=256
CROP_SIZE=2500

echo "=========================================="
echo "GPU Information"
echo "=========================================="
nvidia-smi
echo "=========================================="

echo "Starting SIREN training..."

if [ -f "$FITS_FILE" ]; then
    singularity exec --nv --writable-tmpfs "$CONTAINER" bash -c '
       pip install --user "numpy<2" ptwt >/dev/null 2>&1


        python MilkyWay_siren.py \
            --fits_file "'"$FITS_FILE"'" \
            --crop_size '"$CROP_SIZE"' \
            --sidelength '"$SIDELENGTH"' \
            --hidden_features '"$HIDDEN_FEATURES"' \
            --hidden_layers '"$HIDDEN_LAYERS"' \
            --steps '"$STEPS"' \
            --batch_size_pixels '"$BATCH_SIZE"' \
            --output_dir "'"$OUTPUT_DIR"'" \
            --save_model
    '
else
    singularity exec --nv --writable-tmpfs "$CONTAINER" bash -c '
        pip install --user "numpy<2" ptwt >/dev/null 2>&1

        python MilkyWay_siren.py \
            --sidelength '"$SIDELENGTH"' \
            --hidden_features '"$HIDDEN_FEATURES"' \
            --hidden_layers '"$HIDDEN_LAYERS"' \
            --steps '"$STEPS"' \
            --batch_size_pixels '"$BATCH_SIZE"' \
            --output_dir "'"$OUTPUT_DIR"'" \
            --save_model
    '
fi

echo "=========================================="
echo "Job completed at: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
