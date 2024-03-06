#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=30:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=2

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

srun python3 experiments/cifar/train_conv.py --cluster True --use_residual True --n_epochs 30 --runs 5 --n_hidden_layers 2 --use_wanb True
# srun python3 experiments/cifar/train_conv.py --cluster True --use_residual False --n_epochs 30 --runs 5 --n_hidden_layers 2 --use_wanb True
# srun python3 experiments/cifar/train_conv.py --cluster True --use_residual True --n_epochs 30 --runs 5 --n_hidden_layers 4 --use_wanb True
# srun python3 experiments/cifar/train_conv.py --cluster True --use_residual False --n_epochs 30 --runs 5 --n_hidden_layers 4 --use_wanb True
# srun python3 experiments/cifar/train_conv.py --cluster True --use_residual True --n_epochs 30 --runs 5 --n_hidden_layers 6 --use_wanb True
# srun python3 experiments/cifar/train_conv.py --cluster True --use_residual False --n_epochs 30 --runs 5 --n_hidden_layers 6 --use_wanb True
# srun python3 experiments/cifar/train_conv.py --cluster True --use_residual True --n_epochs 30 --runs 5 --n_hidden_layers 12 –residual_every_n 3 --use_wanb True
# srun python3 experiments/cifar/train_conv.py --cluster True --use_residual False --n_epochs 30 --runs 5 --n_hidden_layers 12 --use_wanb True
