#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=all-nights
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/all_nights_%j.txt
# Step 3. Whole-night generation of all 88 participants with one trained run.
# Run from /project/def-plago/yxtao/dataset-generation-30s-apnea, once per model:
#   sbatch --export=ALL,RUN=/home/yxtao/links/scratch/dataset-generation-30s-apnea/runs/dreamt_30s_apnea_<JOB>,LABEL=30s_apnea submit_generate_nights.sh
#   sbatch --export=ALL,RUN=/home/yxtao/links/scratch/dataset-generation-v2/runs/dreamt_recommended_1346112,LABEL=2min submit_generate_nights.sh
# The run's own workspace/dataset_config.json decides window length (30 s / 2 min) and
# whether the event channel exists. CHECKPOINT defaults to the KL-selected best epoch.
# Participants already finished with the same checkpoint and seed are skipped, so a job
# that hits the 24 h limit can simply be submitted again with the same command.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
module load scipy-stack/2024a
source ./.venv/bin/activate
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONFAULTHANDLER=1

: "${RUN:?set RUN to a training run folder}"
: "${LABEL:?set LABEL, for example 2min or 30s_apnea}"
CHECKPOINT="${CHECKPOINT:-$RUN/training/best_kl_model.pth}"
CONFIG="${CONFIG:-$RUN/workspace/dataset_config.json}"
OUTPUT="${OUTPUT:-/home/yxtao/links/scratch/dataset-generation-30s-apnea/nights/${LABEL}}"
SEED="${SEED:-20261121}"

echo "run=${RUN} checkpoint=${CHECKPOINT} config=${CONFIG} output=${OUTPUT}"
srun python -B -u scripts/generate_all_nights.py --checkpoint "$CHECKPOINT" --config "$CONFIG" \
  --label "$LABEL" --output-dir "$OUTPUT" --seed "$SEED" --threads "${SLURM_CPUS_PER_TASK:-4}"
echo "Nights: $OUTPUT/subjects  (DREAMT-format copies: $OUTPUT/night_npz)"
