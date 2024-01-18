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
module load cuda/12.3
conda activate /home/nfs/adelossantossu/.conda/envs/computerEnv
echo "nvcc version: " `nvcc --version`
echo "python version: " `python --version`
echo "cupy version: " `python -c "import cupy; print(cupy.__version__)"`

srun python experiments/mnist/train_spike_count.py
