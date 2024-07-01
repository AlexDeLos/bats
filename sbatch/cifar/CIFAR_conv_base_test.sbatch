#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=long

#SBATCH --time=90:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "Starting 9 layers No residual MLP on cifar"
srun python3 experiments/cifar/train.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 3 --batch_size 50 --use_wanb True
