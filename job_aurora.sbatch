#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=24:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=4

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu:v100

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myclone
echo "cuda version (from torch): " `python3 -c "import torch; print(torch.version.cuda)"`
echo "cuda version: " `python3 -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"`

srun python3 experiments/mnist/train_spike_count.py