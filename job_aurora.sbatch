#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=30:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

# conda activate /home/nfs/adelossantossu/.conda/envs/freshPytorch
# echo "cuda version (from torch): " `python3 -c "import torch; print(torch.version.cuda)"`
# echo "cuda version: " `python3 -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"`

# srun python3 experiments/mnist/train_spike_count.py
srun python3 experiments/mnist/train_conv.py