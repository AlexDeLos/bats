#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --mail-type=END

#SBATCH --gres=gpu:2

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv
echo "python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_residual True --residual_every_n 4 --residual_jump_length 3 --use_wanb True"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_residual True --residual_every_n 4 --residual_jump_length 3 --use_wanb True
echo "python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_wanb True"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_wanb True