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

echo "emnist -----------------------------------------------------------------------"
echo "Starting 6 layers No residual MLP on emnist"
srun python3 experiments/emnist/train.py --restore True  --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 5 --batch_size 30 --use_residual True --fuse_func Append --use_wanb True --residual_every_n 3 --residual_jump_length 3
