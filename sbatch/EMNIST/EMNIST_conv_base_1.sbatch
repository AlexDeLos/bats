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

echo "emnist_RES -----------------------------------------------------------------------"
echo "Starting 3 layers conv on emnist NO CLUSTER"
srun python3 experiments/emnist/train_conv.py --cluster True --n_epochs 200 --runs 1 --n_hidden_layers 1 --batch_size 40 --use_wanb True 
