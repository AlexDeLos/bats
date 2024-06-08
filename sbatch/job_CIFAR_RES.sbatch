#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "cifar_RES -----------------------------------------------------------------------"
echo "Starting 5 layers residual MLP on cifar"
srun python3 experiments/cifar/train.py --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 4 --batch_size 50 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3
echo "Starting 9 layers residual MLP on cifar"
srun python3 experiments/cifar/train.py --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 9 --batch_size 50 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3
echo "Starting 13 layers residual MLP on cifar"
srun python3 experiments/cifar/train.py --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3