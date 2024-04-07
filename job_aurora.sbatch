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
echo "testing CIFAR10"
srun python3 experiments/mnist/train_spike_count.py -l 0.00005 --cluster True --n_epochs 50 --runs 1 --n_hidden_layers 4 --batch_size 100 --use_residual True --residual_every_n 3 --residual_jump_length 3 --fuse_func Append 