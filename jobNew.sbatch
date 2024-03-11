#!/bin/bash

#SBATCH --partition=general
#SBATCH --qos=short
# SBATCH --time=4:00:00
# SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6144
# SBATCH --mail-type=END
# SBATCH --gres=gpu
echo "start"

# source ~/.bashrc

# module use /opt/insy/modulefiles
# module load miniconda/3.9
echo "loaded"
# module load cuda/11.1
# module load cuda/10.1
#! I wanna work with cuda 11.2
# module load cuda/11.2 -> not use this
#assume we are using cuda 11.8 as it is the output of torch.version.cuda
# conda activate /home/nfs/adelossantossu/.conda/envs/freshPytorch


echo "loeaded env"
# echo "nvcc version: " `nvcc --version`
# conda activate myclone
echo "loaded cuda"
# echo "nvcc version: " `nvcc --version`
# cuda 12.3
echo "cuda version (from torch): " `python3 -c "import torch; print(torch.version.cuda)"` #11.8
# echo "cupy version: " `python3 -c "import cupy; print(cupy.__version__)"`
echo "cuda version: " `python3 -c "import cupy; print(cupy.cuda.runtime.runtimeGetVersion())"`
echo "test ran"

# Find CUDA path

echo "cupy version: " `python3 -c "import cupy; print(cupy.__version__)"`

# srun python3 experiments/mnist/train_spike_count.py
