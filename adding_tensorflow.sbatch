#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=5:00:00

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=1

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu

# source ~/.bashrc

srun conda install conda-forge::tensorflow