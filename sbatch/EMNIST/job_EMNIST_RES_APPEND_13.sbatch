#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=long

#SBATCH --time=168:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "emnist_RES_APPEND -----------------------------------------------------------------------"
echo "Starting 13 layers residual APPEND MLP on emnist"
srun python3 experiments/emnist/train.py  --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 13 --batch_size 15 --use_residual True --fuse_func Append --use_wanb True --residual_every_n 4 --residual_jump_length 3
