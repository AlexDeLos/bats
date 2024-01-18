#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
# SBATCH --time=4:00:00
# SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6144
# SBATCH --mail-type=END
# SBATCH --gres=gpu
# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
# module load cuda/12.3
module use cuda/10.1
conda activate /home/nfs/adelossantossu/.conda/envs/newPytorch
# CONDA_ENV=$(conda info --envs | awk '$1=="*" {print $2}')

# echo "Currently activated Conda environment: $CONDA_ENV"

# echo "conda env list: " `conda env list`
# echo "nvcc version: " `nvcc --version`
# echo "python version: " `python --version`
# echo "pip list: " `pip list`
# echo "conda list: " `conda list`
# echo "cupy version: " `python3 -c "import cupy; print(cupy.__version__)"`

# cuda path = /opt/insy/cuda/12.3/bin/nvcc
# echo %CUDA_PATH%

srun python3 experiments/mnist/train_spike_count.py
