#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=apnea-labels
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/apnea_labels_%j.txt
# Step 1. Respiratory-event sidecars for every DREAMT_PIBI_SE NPZ (no model, no GPU work;
# the GPU allocation is only the usual way to get CPUs on this login node).
# Run from /project/def-plago/yxtao/dataset-generation-30s-apnea:
#   sbatch submit_apnea_labels.sh [RAW_DIR]
# The NPZ folder is only read. Resubmitting keeps finished sidecars.
# Check afterwards: OUT/report.csv (status, event fractions, events per sleep hour vs AHI).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load python/3.11
module load scipy-stack/2024a
source ./.venv/bin/activate
# $HOME and $PROJECT are read-only inside Trillium jobs: keep matplotlib/torch caches on the job's local disk.
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# The raw folder can be given as the first argument (works whatever sbatch does with
# the environment):  sbatch submit_apnea_labels.sh /path/to/dreamt/data_100Hz
# DREAMT 2.1.0, the version DREAMT_PIBI_SE was made from (2.2.0 re-aligned the recordings and does not match).
RAW_DIR="${1:-${RAW_DIR:-/home/yxtao/links/scratch/checkdataset/data_100Hz}}"
# AHI for the sanity check comes from participant_info.csv in the NPZ folder (the file training uses).
if [ ! -d "$RAW_DIR" ]; then
  echo "Raw DREAMT folder not found: $RAW_DIR" >&2
  echo "Find it with: find /scratch/\$USER /project/def-plago -maxdepth 7 -name 'S002_*.csv' 2>/dev/null" >&2
  exit 1
fi
NPZ_DIR="${NPZ_DIR:-/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE}"
OUT="${OUT:-/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE_APNEA}"
VERIFY_IBI_EPOCHS="${VERIFY_IBI_EPOCHS:-0}"   # e.g. 3 recomputes 3 epochs of IBI per subject with neurokit2

srun python -B -u scripts/build_apnea_labels.py --raw-dir "$RAW_DIR" --npz-dir "$NPZ_DIR" \
  --output-dir "$OUT" --workers "${SLURM_CPUS_PER_TASK:-4}" --verify-ibi-epochs "$VERIFY_IBI_EPOCHS"
echo "Sidecars: $OUT  (report: $OUT/report.csv)"
