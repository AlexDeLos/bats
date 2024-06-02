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

echo "emnist_RES -----------------------------------------------------------------------"
echo "Starting 3 layers conv on emnist NO CLUSTER"
srun python3 experiments/mnist/train_conv.py --cluster True --n_epochs 80 --runs 1 --n_hidden_layers 3 --batch_size 50 --use_wanb True --restore True
# echo "Starting 9 layers residual conv on emnist"
# srun python3 experiments/emnist/train_conv.py --n_epochs 50 --runs 1 --n_hidden_layers 9 --batch_size 50 --use_wanb True
# echo "Starting 13 layers residual conv on emnist"
# srun python3 experiments/emnist/train_conv.py --n_epochs 50 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_wanb True