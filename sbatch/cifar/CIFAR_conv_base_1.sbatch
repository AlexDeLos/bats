#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=long

#SBATCH --time=168:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu:v100:2

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "cifar -----------------------------------------------------------------------"
echo "Starting 2 layers No residual conv on cifar"
srun python3 experiments/cifar/train_conv.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 1 --batch_size 20 --use_wanb True 