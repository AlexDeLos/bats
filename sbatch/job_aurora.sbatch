#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu:1

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv
echo "srun python3 experiments/fashion_mnist/train_conv.py --cluster True --n_epochs 30 --runs 1 --n_hidden_layers 4 --batch_size 40 --use_residual True --use_wanb True --residual_every_n 3 --residual_jump_length 3 "
srun python3 experiments/fashion_mnist/train_conv.py --cluster True --n_epochs 30 --runs 1 --n_hidden_layers 4 --batch_size 40 --use_residual True --use_wanb True --residual_every_n 3 --residual_jump_length 3 