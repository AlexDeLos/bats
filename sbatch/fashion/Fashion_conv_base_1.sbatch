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
srun python3 experiments/fashion_mnist/train_conv.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 1 --batch_size 30 --use_wanb True