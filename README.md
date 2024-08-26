# Residual Connections in Spiking Neural Networks (Res-BATS)

This is the repository containing the source code for the Residual Connections in Spiking Neural Networks (Res-BATS) [1] which is part of the author's master's thesis. This repository builds upon Error Backpropagation Through Spikes (BATS) [2] is a GPU-compatible algorithm that extends Fast & Deep [3], a method to performs exact gradient descent in Deep Spiking Neural Networks (SNNs).

In contrast with Fast & Deep, BATS allows error backpropagation with multiple spikes per neuron, leading to increased 
performances. The proposed algorithm backpropagates the errors through post-synaptic spikes with linear time complexity 
<em>O(N)</em> making the error backpropagation process fast for multi-spike SNNs.<br>

In contrast with BATS, this residual approach allows for the use of deeper networks thanks to the addition of residual connections. This is done by using an architecture based on fuse functions that can allow for these functions to be exchangable in order to take andvantage of the flexilibty provided by precise time and multi sppiking properties of the netowrk.

This repository contains the full Cuda implementations of our efficient event-based SNN simulator and the BATS algorithm.
All experiments used in the development and paper in order to allow for result reproduction. The experiments are ran on the MNIST, EMNIST, Fashion MNIST and CIFAR-10 datasets.


## Dependencies and Libraries

Recommanded Python version: >= 3.8

Libraries:
- Cuda (we suggest Cuda 12.3 as this is the version that we used to develop Res-BATS 
  but other versions should also work)
  
Python packages:
- CuPy [4] (corresponding to the installed version of Cuda)
- matplotlib (<em>Optional</em>. Install only if generate plots with monitors)
- requests (<em>Optional</em>. Install only if run the scripts to download the 
  experiments' datasets)
- scipy
- wandb (<em>Optional</em>. Install only if you wish to record the results using Weights and Biases (https://wandb.ai/))

## Experiments

In the sbatch folder one can see the multiple tests used in the development of the paper, however one can run the test with their of hyperparamaters by adding arguments to the Python script call. The default values of the hyperparamaters aare found in the "bats\Utils\utils.py" file.

### Residual experiments
In the following commands a 6 layer residual network will be ran for 200 epochs.
```console
$ python3 experiments/mnist/train.py --cluster True --use_residual True

$ python3 experiments/mnist/train_conv.py --cluster True --use_residual True

$ python3 experiments/emnist/train.py --cluster True --use_residual True

$ python3 experiments/emnist/train_conv.py --cluster True --use_residual True

$ python3 experiments/fashion_mnist/train.py --cluster True --use_residual True

$ python3 experiments/fashion_mnist/train_conv.py --cluster True --use_residual True

$ python3 experiments/cifar/train.py --cluster True --use_residual True

$ python3 experiments/cifar/train_conv.py --cluster True --use_residual True
```

Running these python scripts can take a long time depending on the arguments used and the hardware the code is running in. For deeper than 9 layers in MLP or 3 layers in CNN we recomend the use of a computer cluster.

### Weights and Biases

This repository uses https://wandb.ai/ in order to record many diferent metrics on each run. This can be activated by using the ```--use_wanb True``` flag.

For information on how to set an account up follow: https://docs.wandb.ai/quickstart


### Download datasets

#### MNIST

```console
$ cd datasets
$ python3 get_mnist.py
Downloading MNIST...
[██████████████████████████████████████████████████]
Done.
```

#### EMNIST

```console
$ cd datasetsget_mnist.py
$ python3 get_emnist.py
Downloading EMNIST...
[██████████████████████████████████████████████████]
Done.
Extracting EMNIST...
Done.
Cleaning...
Done.
```
Downloading the EMNIST dataset may take a few minutes due to the size of the file.

#### Fashion MNIST

```console
$ cd datasets
$ python3 get_fashion_mnist.py
Downloading Fashion MNIST...
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
Done.
```

#### CIFAR

in the case you wish to use cifar 100 replace the 10 with 100.
```console
$ cd datasetsget_mnist.py
$ python3 get_cifar10.py
Downloading cifar10...
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
[██████████████████████████████████████████████████]
Done.
```

### Train models recomentations
MLP models will be trained faster than their CNN counterparts. If you are suffering form techical limitations due to the GPU in your machine we would recomend the following:
 - Use shallower Networks
 - Use lighter layers: less neurons per layer (```--n_neurons```) in the case of MLP and reducing the size and amount of channels (```--cnn_channels```) of the kernels in the CNN implementations.
 - Use a subset of the database.

## References
[2] A. De Los Santos Subirats (2024). Residual Connections in Spiking Neural Networks. https://repository.tudelft.nl/record/uuid:7b42e9fb-f47c-46bc-a125-c481a59fc007 <br>
[2] Bacho, F., & Chu, D.. (2022). Exact Error Backpropagation Through Spikes for Precise Training of Spiking Neural Networks. https://arxiv.org/abs/2212.09500 <br>
[3] J. Göltz, L. Kriener, A. Baumbach, S. Billaudelle, O. Breitwieser, B. Cramer, D. Dold, A. F. Kungl, W. Senn, J. Schemmel, K. Meier, & M. A. Petrovici (2021). Fast and energy-efficient neuromorphic deep learning with first-spike times. <em>Nature Machine Intelligence, 3(9), 823–835.</em> <br>
[4] Okuta, R., Unno, Y., Nishino, D., Hido, S., & Loomis, C. (2017). CuPy: A NumPy-Compatible Library for NVIDIA GPU Calculations. In <em>Proceedings of Workshop on Machine Learning Systems (LearningSys) in The Thirty-first Annual Conference on Neural Information Processing Systems (NIPS).</em> <br>
[5] Stimberg, M., Brette, R., & Goodman, D. (2019). Brian 2, an intuitive and efficient neural simulator. <em>eLife, 8, e47314.<em>