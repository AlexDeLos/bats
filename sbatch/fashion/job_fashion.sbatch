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

echo "fashion_mnist -----------------------------------------------------------------------"
echo "Starting 1 layers No residual MLP on fashion_mnist"
srun python3 experiments/fashion_mnist/train.py --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 1 --batch_size 100 --use_wanb True
echo "Starting 4 layers No residual MLP on fashion_mnist"
srun python3 experiments/fashion_mnist/train.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 4 --batch_size 100 --use_wanb True
# echo "Starting 9 layers No residual MLP on fashion_mnist"
# srun python3 experiments/fashion_mnist/train.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 9 --batch_size 50 --use_wanb True
# echo "Starting 13 layers No residual MLP on fashion_mnist"
# srun python3 experiments/fashion_mnist/train.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_wanb True