#!/bin/bash
#SBATCH --job-name=mvtcr_ig
#SBATCH --output=ig_%A_%a.log
#SBATCH --error=ig_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --array=0-1

#  mvTCR Integrated Gradients; both RNA baselines, one array job
#   task 0 : --baseline zero   x' = 0
#   task 1 : --baseline mean   x' = mean training expression

set -o pipefail

case "${SLURM_ARRAY_TASK_ID}" in
    0) BASELINE="zero" ;;
    1) BASELINE="mean" ;;
    *) echo "unexpected array index ${SLURM_ARRAY_TASK_ID}"; exit 1 ;;
esac

echo " mvTCR IG analysis  |  baseline = ${BASELINE}"
echo "started   : $(date)"
echo "job       : ${SLURM_ARRAY_JOB_ID}  task ${SLURM_ARRAY_TASK_ID}"
echo "node      : $(hostname)"
echo "partition : ${SLURM_JOB_PARTITION}"
echo "gpus      : ${CUDA_VISIBLE_DEVICES}"
echo

nvidia-smi || echo "WARNING: nvidia-smi failed -- is this a GPU node?"
echo

MVTCR_ENV="${MVTCR_ENV:-mvTCR}"
MVTCR_DIR="${MVTCR_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"

module load Miniforge3 2>/dev/null || true
conda activate "${MVTCR_ENV}" || {
    echo "!!! could not activate '${MVTCR_ENV}' -- set MVTCR_ENV"; exit 1; }

cd "${MVTCR_DIR}" || {
    echo "!!! could not cd into '${MVTCR_DIR}' -- set MVTCR_DIR"; exit 1; }

echo "python : $(which python)"
echo "env    : ${CONDA_PREFIX}"
echo

# Unit tests first. 33 checks covering IG completeness, class-median normalisation,
# baseline equivalence, and the embedding-injection plumbing. If the maths is
# wrong there is no point spending GPU hours finding out slowly.

echo "--- unit tests ---"
python test_attribution_core.py
if [ $? -ne 0 ]; then
    echo "!!! unit tests FAILED -- aborting before the expensive run"
    exit 1
fi
echo

# Analysis:
echo "--- analysis (baseline: ${BASELINE}) ---"
python 01_run_analysis.py \
    --baseline "${BASELINE}" \
    --splits 0 1 2 3 4 \
    --n-steps 128 \
    --cell-batch 64 \
    --step-chunk 8 \
    --top-n 20
RC=$?

echo
if [ $RC -ne 0 ]; then
    echo "!!! 01_run_analysis.py exited ${RC} !!!"
    echo "Per-split .npz files and figures written before the failure are"
    echo "still on disk. Rerun only the missing splits with --splits <n>."
else
    echo "=== ${BASELINE} baseline finished cleanly at $(date) ==="
fi

echo
echo "results : ../results/interpretability/${BASELINE}/"
echo "figures : ../figures/interpretability/${BASELINE}/"
echo
echo "When BOTH array tasks are done, compare them:"
echo "    python 02_compare_baselines.py 2>&1 | tee comparison.txt"

exit $RC
