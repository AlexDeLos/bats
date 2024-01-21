#!/bin/sh

# SBATCH --partition=general
# SBATCH --qos=short
# SBATCH --time=0:02:00
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=4
# SBATCH --mem=6144
# SBATCH --mail-type=END
# SBATCH --gres=gpu
# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
# module load cuda/12.2
# module load cuda/10.1
#!use this because for some reason the default
# echo "nvcc version: " `nvcc --version`
conda activate /home/nfs/adelossantossu/.conda/envs/pytorch
#fresh py gives circular import error
# echo "python version: " `python --version`
# CONDA_ENV=$(conda info --envs | awk '$1=="*" {print $2}')
# echo "Currently activated Conda environment: $CONDA_ENV"

echo "cupy version: " `python3 -c "import cupy; import sys; print(sys.version); print(cupy.__version__)"`

# echo "conda env list: " `conda env list`
# echo "nvcc version: " `nvcc --version`
# echo "pip list: " `pip list`
# echo "conda list: " `conda list`

# cuda path = /opt/insy/cuda/12.3/bin/nvcc
# echo %CUDA_PATH%

# srun python3 experiments/mnist/train_spike_count.py
