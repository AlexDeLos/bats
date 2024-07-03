#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=long

#SBATCH --mail-type=END

#SBATCH --time=168:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "mnist_RES -----------------------------------------------------------------------"
echo "Starting 5 layers residual conv on emnist"
srun python3 experiments/mnist/train_conv.py --slope_decay True --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 4 --batch_size 20 --use_residual True --residual_every_n 3 --residual_jump_length 3 --use_wanb True 
# echo "Starting 9 layers residual conv on emnist"
# srun python3 experiments/emnist/train_conv.py --n_epochs 200 --runs 1 --n_hidden_layers 9 --batch_size 50 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3
# echo "Starting 13 layers residual conv on emnist"
# srun python3 experiments/emnist/train_conv.py --n_epochs 200 --runs 1 --n_hidden_layers 13 --batch_size 50 --use_residual True --use_wanb True --residual_every_n 4 --residual_jump_length 3