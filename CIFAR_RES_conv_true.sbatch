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

echo "cifar_RES -----------------------------------------------------------------------"
echo "Starting 5 layers residual conv on cifar"
srun python3 experiments/cifar/train_conv.py --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 4 --batch_size 30 --use_residual True --use_wanb True --residual_every_n 3 --residual_jump_length 3 --restore True
echo "Starting 9 layers residual conv on cifar"
srun python3 experiments/cifar/train_conv.py  --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 9 --batch_size 30 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3
# echo "Starting 13 layers residual conv on cifar"
# srun python3 experiments/cifar/train_conv.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 13 --batch_size 20 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3