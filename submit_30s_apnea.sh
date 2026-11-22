#!/bin/bash
#SBATCH --account=def-plago
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --job-name=30s-apnea
#SBATCH --output=/home/yxtao/links/scratch/dataset-generation/logs/30s_apnea_%j.txt
# Step 2. Train the 30-second model with the respiratory-event condition (11 channels).
# Run from /project/def-plago/yxtao/dataset-generation-30s-apnea after step 1:
#   sbatch submit_30s_apnea.sh                                        # 30 s + events (default)
#   sbatch --export=ALL,CONFIG=configs/dreamt_30s.json submit_30s_apnea.sh   # 30 s, no events (ablation)
#   sbatch --dependency=afterok:<labels job id> submit_30s_apnea.sh
# Same settings as configs/dreamt_recommended.json (preserve, [-1, 1] min-max on 0-2 s,
# S4 dropout 0.03, EMA 0.9999, KL checkpoint selection on S092) except window_epochs=1
# and the event channel. Trillium jobs are at most 24 h; run_dreamt.py cannot resume.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
module load StdEnv/2023 gcc/12.3
module load opencv/4.9.0
module load python/3.11
module load scipy-stack/2024a   # numpy 1.26.4 / pandas / matplotlib come from this module
source ./.venv/bin/activate
# $HOME and $PROJECT are read-only inside Trillium jobs: keep matplotlib/torch caches on the job's local disk.
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp}/cache" MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/matplotlib"
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# Set before Python imports NumPy/PyTorch (same thread policy as the earlier runs).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export BLIS_THREAD_IMPL=single BLIS_NUM_THREADS=1
export BLIS_JC_NT=1 BLIS_PC_NT=1 BLIS_IC_NT=1 BLIS_JR_NT=1 BLIS_IR_NT=1
export PYTHONFAULTHANDLER=1

EPOCHS="${EPOCHS:-50}"
CONFIG="${CONFIG:-configs/dreamt_30s_apnea.json}"
RUN_NAME="$(basename "$CONFIG" .json)"
OUTPUT="/home/yxtao/links/scratch/dataset-generation-30s-apnea/runs/${RUN_NAME}_${SLURM_JOB_ID}"   # scratch: writable in jobs

python -c "import sys,torch,numpy;print('env',sys.version.split()[0],torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),numpy.__version__,torch.cuda.get_device_name(0))"
echo "epochs=${EPOCHS} config=${CONFIG} output=${OUTPUT}"

srun python -B -u scripts/run_dreamt.py \
  --config "$CONFIG" --output-dir "$OUTPUT" \
  --epochs "$EPOCHS" --batch-size 8 --threads "${SLURM_CPUS_PER_TASK:-4}"

printf 'Completed 30-s run (%s epochs). Outputs: %s\n' "$EPOCHS" "$OUTPUT"
