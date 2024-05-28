#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --mail-type=END

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "EMNIST_RES -----------------------------------------------------------------------"
echo "Starting 4 layers residual MLP on EMNIST"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 4 --batch_size 80 --use_residual True --use_wanb True --residual_every_n 3 --residual_jump_length 3
echo "Starting 9 layers residual MLP on EMNIST"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 9 --batch_size 80 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3
echo "Starting 13 layers residual MLP on EMNIST"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 100 --runs 1 --n_hidden_layers 13 --batch_size 80 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3