#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=4

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "EMNIST_RES -----------------------------------------------------------------------"
echo "Starting 4 layers residual MLP on EMNIST"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 4 --batch_size 50 --use_residual True --use_wanb True
echo "Starting 6 layers residual MLP on EMNIST"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 6 --batch_size 50 --use_residual True --use_wanb True
echo "Starting 12 layers residual MLP on EMNIST"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 12 --batch_size 50 --use_residual True --use_wanb True