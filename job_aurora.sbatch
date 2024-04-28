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
echo "python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 9 --batch_size 100 --use_residual True --residual_every_n 3 --residual_jump_length 2 --use_wanb True --fuse_func Append"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 9 --batch_size 100 --use_residual True --residual_every_n 3 --residual_jump_length 2 --use_wanb True --fuse_func Append