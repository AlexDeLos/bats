#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=long

#SBATCH --time=60:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "mnist_RES_Average -----------------------------------------------------------------------"
echo "Starting 9 layers residual Average MLP on mnist"
srun python3 experiments/mnist/train.py  --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 9 --batch_size 80 --use_residual True --fuse_func Average --use_wanb True --residual_every_n 4 --residual_jump_length 3 
