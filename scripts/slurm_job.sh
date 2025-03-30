#!/bin/bash
#SBATCH --job-name=sign_language_detection
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Sign Language Detection SLURM Job
# -------------------------------
# This script runs the sign language detection pipeline on a SLURM-managed cluster.

# Load required modules (adjust based on your cluster environment)
# module load python/3.8
# module load cuda/11.3
# module load cudnn/8.2.0

# Activate virtual environment if needed
# source /path/to/venv/bin/activate

# Directory setup
SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$(realpath "$SCRIPT_DIR/..")
cd "$PROJECT_DIR" || exit 1

# Create log directory if it doesn't exist
mkdir -p logs

# Parse arguments
VIDEO_PATH="$1"
OUTPUT_DIR="$2"
USE_SEG=""

if [ "$3" == "--seg" ]; then
    USE_SEG="--seg"
fi

# Check required arguments
if [ -z "$VIDEO_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Usage: sbatch $0 <video_path> <output_dir> [--seg]"
    echo "  --seg    Use segmentation (optional)"
    exit 1
fi

# Print job information
echo "====== Job Information ======"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "Project directory: $PROJECT_DIR"
echo "Video path: $VIDEO_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Using segmentation: ${USE_SEG:-false}"
echo "============================"

# Run the pipeline script
bash "$PROJECT_DIR/scripts/run_pipeline.sh" --input "$VIDEO_PATH" --output "$OUTPUT_DIR" $USE_SEG

# Print job completion
echo "====== Job Completed ======"
echo "End time: $(date)"
echo "============================" 