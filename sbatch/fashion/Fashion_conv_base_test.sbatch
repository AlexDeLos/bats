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
echo "Starting 3 layers conv on fashion_mnist NO CLUSTER"
srun python3 experiments/fashion_mnist/train_conv.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 3 --batch_size 30 --use_wanb True  --use_residual True --residual_every_n 2 --residual_jump_length 2 # 
# echo "Starting 9 layers residual conv on fashion_mnist"
# srun python3 experiments/fashion_mnist/train_conv.py --n_epochs 200 --runs 1 --n_hidden_layers 9 --batch_size 50 --use_wanb True
# echo "Starting 13 layers residual conv on fashion_mnist"
# srun python3 experiments/fashion_mnist/train_conv.py --n_epochs 200 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_wanb True