#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384
#SBATCH --mail-type=END
#SBATCH --gres=gpu


module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/12.3
conda activate /home/nfs/adelossantossu/.conda/envs/pytorch
export CUPY_GPU_MEMORY_DEBUG=1

srun python experiments/mnist/train_spike_count.py
