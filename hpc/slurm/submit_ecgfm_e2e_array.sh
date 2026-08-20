#!/bin/bash
# Stanage SLURM array submission for Model C (random-init ECG-FM, end-to-end, 10-fold CV).
# One GPU per fold. 10 concurrent tasks fit under the 12-GPU user limit.
#
# Before first submission:
#   1. Replace <your-username> below with your Sheffield HPC username.
#   2. Replace <your-email> with your university email.
#   3. dos2unix this file (Windows line endings = silent SLURM breakage).
#   4. Confirm layout on parscratch:
#        /mnt/parscratch/users/$USER/diss/data/            (4 data files)
#        /mnt/parscratch/users/$USER/diss/hpc/             (all training scripts, _hpc suffix)
#        /mnt/parscratch/users/$USER/diss/ecg_fm_checkpoints/  (.pt architecture source)
#        /mnt/parscratch/users/$USER/diss/fairseq-signals/ (cloned + pip install -e .)
#   5. Confirm `ssl_ecg` conda env exists on cluster.
#
# Submission (run from $DISS_ROOT or supply full path):
#   sbatch /mnt/parscratch/users/$USER/diss/hpc/slurm/submit_ecgfm_e2e_array.sh

#SBATCH --job-name=ecgfm_e2e
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=15:00:00
#SBATCH --array=1-10
#SBATCH --output=/mnt/parscratch/users/%u/diss/logs/ecgfm_e2e_%A_%a.out
#SBATCH --error=/mnt/parscratch/users/%u/diss/logs/ecgfm_e2e_%A_%a.err
#SBATCH --mail-user=<your-email>
#SBATCH --mail-type=FAIL,END

set -euo pipefail

module purge
module load Anaconda3/2022.05
module load CUDA/12.1.1
export PATH=/mnt/parscratch/users/$USER/conda/envs/ssl_ecg/bin:$PATH


DISS_ROOT=/mnt/parscratch/users/$USER/diss
cd "$DISS_ROOT"
export PYTHONPATH="$DISS_ROOT/hpc"
source ~/.wandb_key
export WANDB_MODE=disabled
export WANDB_DIR=$DISS_ROOT/wandb_cache
mkdir -p $DISS_ROOT/wandb_cache

mkdir -p logs

echo "[`date`] fold=$SLURM_ARRAY_TASK_ID  job=$SLURM_JOB_ID  node=$(hostname)"
nvidia-smi

python "$DISS_ROOT/hpc/master_ecgfm_e2e_hpc.py" \
    --config_dir "$DISS_ROOT/hpc/ecgfm_e2e_config.yaml" \
    --fold_index "$SLURM_ARRAY_TASK_ID"

echo "[`date`] fold=$SLURM_ARRAY_TASK_ID done"
