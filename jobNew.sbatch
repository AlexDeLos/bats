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

module use /opt/insy/modulefiles
module load miniconda/3.9
echo "loaded"
# module load cuda/11.1
# module load cuda/10.1
#! I wanna work with cuda 11.2
# module load cuda/11.2
conda activate /home/nfs/adelossantossu/.conda/envs/freshPytorch
echo "loeaded env"
# cuda 12.3
echo "cupy version: " `python3 -c "import cupy; print(cupy.__version__)"`
echo "test ran"

# Find CUDA path
# CUDA_PATH=$(dirname $(dirname $(which nvcc)))

echo "nvcc version: " `nvcc --version`
# echo "python version: " `python --version`
echo "pip list: " `pip list`
echo "conda list: " `conda list`
# echo "CUDA_PATH: " $CUDA_PATH

# # Set LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

echo "cupy version: " `python3 -c "import cupy; print(cupy.__version__)"`

# srun python3 experiments/mnist/train_spike_count.py
