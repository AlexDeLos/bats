#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=32:00:00

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=4

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu:2

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv
echo "python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 5 --batch_size 100 --use_residual True --use_wanb True"
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 5 --batch_size 100 --use_residual True --use_wanb True