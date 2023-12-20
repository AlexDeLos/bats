#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8192
#SBATCH --mail-type=END
#SBATCH --gres=gpu


module use /opt/insy/modulefiles
module load miniconda/3.9
module load cuda/12.3
conda activate /home/nfs/adelossantossu/.conda/envs/pytorch
conda list >> conda_list.out
nvcc --version >> nvcc_version.out

srun python experiments/mnist/train_spike_count.py
