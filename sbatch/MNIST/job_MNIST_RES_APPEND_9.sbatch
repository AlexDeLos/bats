#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=long

#SBATCH --mail-type=END

#SBATCH --time=60:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=8384

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "EMNIST_RES_APPEND -----------------------------------------------------------------------"
echo "Starting 13 layers residual APPEND MLP on EMNIST"
srun python3 experiments/mnist/train.py  --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 9 --batch_size 80 --use_residual True --fuse_func Append --use_wanb True --residual_every_n 4 --residual_jump_length 3
