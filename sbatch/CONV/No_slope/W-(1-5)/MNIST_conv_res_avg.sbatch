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
srun python3 experiments/mnist/train_conv.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 4 --batch_size 20 --use_residual True --residual_every_n 3 --residual_jump_length 3 --use_wanb True --fuse_func Average --w1 1 --w2 5