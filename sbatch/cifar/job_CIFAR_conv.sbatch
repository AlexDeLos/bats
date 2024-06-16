#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu:v100:2

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "cifar -----------------------------------------------------------------------"
# echo "Starting 1 layers No residual conv on cifar"
# srun python3 experiments/cifar/train_conv.py  --restore True --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 1 --batch_size 30 --use_wanb True
echo "Starting 2 layers No residual conv on cifar"
srun python3 experiments/cifar/train_conv.py  --restore True --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 3 --batch_size 20 --use_wanb True 
# echo "Starting 5 layers No residual conv on cifar"
# srun python3 experiments/cifar/train_conv.py  --restore True --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 4 --batch_size 30 --use_wanb True
# echo "Starting 9 layers No residual conv on cifar"
# srun python3 experiments/cifar/train_conv.py  --restore True --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 9 --batch_size 60 --use_wanb True
# echo "Starting 13 layers No residual conv on cifar"d
# srun python3 experiments/cifar/train_conv.py  --restore True --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 13 --batch_size 10 --use_wanb True