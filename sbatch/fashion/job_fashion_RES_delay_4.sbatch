#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=long

#SBATCH --mail-type=END

#SBATCH --time=100:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "fashion_mnist_RES -----------------------------------------------------------------------"
echo "Starting 4 layers residual MLP on fashion_mnist"
srun python3 experiments/fashion_mnist/train.py  --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 4 --batch_size 80 --use_residual True --use_wanb True --residual_every_n 3 --residual_jump_length 3 --fuse_func Append --use_delay True 
