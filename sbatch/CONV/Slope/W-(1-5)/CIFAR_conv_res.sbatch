#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=long

#SBATCH --time=168:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu:1

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "cifar_RES -----------------------------------------------------------------------"
echo "Starting 4 layers residual conv on cifar"
srun python3 experiments/cifar/train_conv.py --slope_decay True --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 4 --batch_size 10 --use_residual True --use_wanb True --residual_every_n 3 --residual_jump_length 3 --w1 1 --w2 5
