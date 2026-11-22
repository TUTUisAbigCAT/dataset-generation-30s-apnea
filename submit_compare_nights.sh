#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --job-name=compare-nights
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/compare_nights_%j.txt
# Step 4. Compare the whole-night generations (CPU only; first run = baseline).
#   sbatch submit_compare_nights.sh
#   sbatch --export=ALL,EXTRA="--run 30s=/home/yxtao/links/scratch/dataset-generation-30s-apnea/nights/30s" submit_compare_nights.sh

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load python/3.11
module load scipy-stack/2024a
source ./.venv/bin/activate
# $HOME and $PROJECT are read-only inside Trillium jobs: keep matplotlib/torch caches on the job's local disk.
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

NIGHTS="${NIGHTS:-/home/yxtao/links/scratch/dataset-generation-30s-apnea/nights}"
BASELINE="${BASELINE:-$NIGHTS/2min}"
NEW="${NEW:-$NIGHTS/30s_apnea}"
LABELS="${LABELS:-/home/yxtao/links/scratch/WatchSleepProcessedDataset/DREAMT_PIBI_SE_APNEA}"
OUTPUT="${OUTPUT:-$NIGHTS/compare_2min_vs_30s_apnea}"

srun python -B -u scripts/compare_night_generations.py --run "2min=$BASELINE" --run "30s_apnea=$NEW" \
  ${EXTRA:-} --apnea-label-dir "$LABELS" --output-dir "$OUTPUT"
echo "Report: $OUTPUT/report.md  figures: $OUTPUT/*.png"
