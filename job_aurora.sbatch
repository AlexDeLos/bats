#!/bin/sh

#SBATCH --partition=general

#SBATCH --qos=medium

#SBATCH --time=30:00:00

#SBATCH --ntasks=2

#SBATCH --cpus-per-task=4

#SBATCH --mem=16384

#SBATCH --mail-type=END

#SBATCH --gres=gpu

# source ~/.bashrc

module use /opt/insy/modulefiles
module load miniconda/3.9
conda activate myHomeEnv

echo "Starting job -----------------------------------------------------------------------"
echo "Starting 1 layers No residual MLP on MNIST"
srun python3 experiments/mnist/train_spike_count.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 1 --batch_size 70 --use_wanb True
echo "Starting 2 layers No residual MLP on MNIST"
srun python3 experiments/mnist/train_spike_count.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 2 --batch_size 70 --use_wanb True
echo "Starting 4 layers No residual MLP on MNIST"
srun python3 experiments/mnist/train_spike_count.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 4 --batch_size 70 --use_wanb True
# echo "Starting 6 layers No residual MLP on MNIST"
# srun python3 experiments/mnist/train_spike_count.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 6 --batch_size 70 --use_wanb True
# echo "Starting 12 layers No residual MLP on MNIST"
# srun python3 experiments/mnist/train_spike_count.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 12 --batch_size 70 --use_wanb True

# echo "NOW ON TO EMNIST -----------------------------------------------------------------------"
# echo "Starting 1 layers No residual MLP on EMNIST"
# srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 1 --batch_size 70 --use_wanb True
# echo "Starting 2 layers No residual MLP on EMNIST"
# srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 2 --batch_size 70 --use_wanb True
# echo "Starting 4 layers No residual MLP on EMNIST"
# srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 4 --batch_size 70 --use_wanb True
# echo "Starting 6 layers No residual MLP on EMNIST"
# srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 6 --batch_size 70 --use_wanb True
# echo "Starting 12 layers No residual MLP on EMNIST"
# srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 12 --batch_size 70 --use_wanb True


# echo 'Starting 12 layers No residual MLP on EMNIST'
# srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 12 --batch_size 20 --use_wanb True
echo 'Starting 2 layers No residual MLP on EMNIST'
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 2 --batch_size 20 --use_wanb True
echo 'Starting 6 layers No residual MLP on EMNIST'
srun python3 experiments/emnist/train.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 6 --batch_size 90 --use_wanb True
# echo "Done with EMNIST -----------------------------------------------------------------------"
# echo "Done with residual Fasle, n_hidden_layers 6 -----------------------------------------------------------------------"
# srun python3 experiments/cifar/train_conv.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 6 --batch_size 20 --test_batch_size 20 --use_wanb True
# # echo "Done with residual True, n_hidden_layers 4 -----------------------------------------------------------------------"
# # srun python3 experiments/cifar/train_conv.py --cluster True --use_residual True --n_epochs 50 --runs 2 --n_hidden_layers 4 --batch_size 20 --test_batch_size 20 --use_wanb True
# # echo "Done with residual False, n_hidden_layers 4 -----------------------------------------------------------------------"
# # srun python3 experiments/cifar/train_conv.py --cluster True --n_epochs 50 --runs 2 --n_hidden_layers 4 --batch_size 20 --test_batch_size 20 --use_wanb True
