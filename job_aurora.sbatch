#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=4

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv
echo "python3 experiments/fashion_mnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 3 --batch_size 50 --use_residual True --residual_jump_length 65 --residual_every_n 2 --fuse_func Append"
srun python3 experiments/fashion_mnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 3 --batch_size 50 --use_residual True --residual_jump_length 65 --residual_every_n 2 --fuse_func Append