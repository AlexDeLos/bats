#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv
echo "python3 experiments/emnist/train_conv.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 3 --batch_size 20 --use_residual True --residual_every_n 2 --residual_jump_length 2 --use_wanb True"
srun python3 experiments/emnist/train_conv.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 3 --batch_size 20 --use_residual True --residual_every_n 2 --residual_jump_length 2 --use_wanb True