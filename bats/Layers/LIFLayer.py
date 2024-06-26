from typing import Callable, Tuple
from typing import Optional
import cupy as cp
import numpy as np

from bats.AbstractLayer import AbstractLayer
from bats.CudaKernels.Wrappers.Inference import *
from bats.CudaKernels.Wrappers.Backpropagation import *


class LIFLayer(AbstractLayer):
    def __init__(self, previous_layer: AbstractLayer, tau_s: float, theta: float, delta_theta: float,
                 weight_initializer: Callable[[int, int], cp.ndarray] = None, max_n_spike: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.__previous_layer: AbstractLayer = previous_layer
        self.__tau_s: cp.float32 = cp.float32(tau_s)
        self.__tau: cp.float32 = cp.float32(2 * tau_s)
        self.__theta_tau: cp.float32 = cp.float32(theta / self.__tau)
        self.__delta_theta_tau: cp.float32 = cp.float32(delta_theta / self.__tau)
        if weight_initializer is None:
            self.__weights: cp.ndarray = cp.zeros((self.n_neurons, previous_layer.n_neurons), dtype=cp.float32)
        else:
            self.__weights: cp.ndarray = weight_initializer(self.n_neurons, previous_layer.n_neurons)
            # self.__weights: cp.ndarray = cp.zeros((self.n_neurons, previous_layer.n_neurons), dtype=cp.float32)
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__a: Optional[cp.ndarray] = None
        self.__x: Optional[cp.ndarray] = None
        self.__post_exp_tau: Optional[cp.ndarray] = None

        self.__pre_exp_tau_s: Optional[cp.ndarray] = None
        self.__pre_exp_tau: Optional[cp.ndarray] = None
        #! this is never used
        # self.__pre_spike_weights: Optional[cp.ndarray] = None
        self.__c: Optional[cp.float32] = self.__theta_tau

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        test = self.__n_spike_per_neuron
        return self.__spike_times_per_neuron, self.__n_spike_per_neuron

    @property
    def weights(self) -> Optional[cp.ndarray]:
        return self.__weights

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        self.__weights = cp.array(weights, dtype=cp.float32)

    def reset(self) -> None:
        self.__n_spike_per_neuron = None
        self.__spike_times_per_neuron = None
        self.__pre_exp_tau_s = None
        self.__pre_exp_tau = None
        # self.__pre_spike_weights = None
        self.__a = None
        self.__x = None
        self.__post_exp_tau = None

    def forward(self, max_simulation: float, training: bool = False) -> None:
        #? What are these two variables?
        #? How is it per neuron? (50, 784, 1)=>shape of spike_times_per_neuron on first stop of debugger
        #? It seems to be the number of spikes per layer, not per neuron, as there where 784 neurons in the input layer
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__previous_layer.spike_trains
        # print(self.name)
        # print(pre_n_spike_per_neuron)
        # print(cp.where(pre_n_spike_per_neuron!=0))
        # if cp.any(cp.isnan(pre_spike_per_neuron)):
        #     raise ValueError("NaNs in pre_spike_per_neuron")


        self.__pre_exp_tau_s, self.__pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)
        # END OF PREVIOUS LAYER INPUTS

        # Sort spikes for inference
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(pre_spike_per_neuron,
                                                                                    pre_n_spike_per_neuron)
        if sorted_indices.size == 0:  # No input spike in the batch
            batch_size = pre_spike_per_neuron.shape[0]
            shape = (batch_size, self.n_neurons, self.__max_n_spike)
            self.__n_spike_per_neuron = cp.zeros((batch_size, self.n_neurons), dtype=cp.int32)
            self.__spike_times_per_neuron = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__post_exp_tau = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__a = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__x = cp.full(shape, cp.inf, dtype=cp.float32)
        else:
            #? What are they sorting the spikes by? TIME obviously
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)

            # Store the times of the spikes (this array A[i]<A[i+1] for all i)
            sorted_spike_times[sorted_indices == -1] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_s, new_shape), sorted_indices, axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(self.__pre_exp_tau, new_shape), sorted_indices, axis=1)
            pre_spike_weights = get_spike_weights(self.weights, sorted_spike_indices)

            # Compute spikes, everything has been calculated in order to make this
            #! nans in self.__spike_times_per_neuron-> there are no nans in the input
            self.__n_spike_per_neuron, self.__a, self.__x, self.__spike_times_per_neuron, \
            self.__post_exp_tau = compute_spike_times(sorted_spike_times, sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                      pre_spike_weights, self.__c,
                                                      self.__delta_theta_tau,
                                                      self.__tau, cp.float32(max_simulation), self.__max_n_spike)
            spikes = self.__spike_times_per_neuron
            count = self.__n_spike_per_neuron
            test3 = self.__post_exp_tau
            # count = self.__n_spike_per_neuron
            # print(self.name)
            # print(cp.where(count!=0)[0].shape)
            # if cp.any(cp.isnan(spikes)):
            #     raise ValueError("NaNs in spikes")

    def backward(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        # if cp.any(cp.isnan(errors)):
        #     raise ValueError("NaNs in errors")
        pre_spike_per_neuron, _ = self.__previous_layer.spike_trains
        propagate_recurrent_errors(self.__x, self.__post_exp_tau, errors, self.__delta_theta_tau)# all the shape of this layer
        f1, f2 = compute_factors(self.__spike_times_per_neuron, self.__a, self.__c, self.__x,
                                 self.__post_exp_tau, self.__tau)
        #! nans show up here when getting the previous layer spikes
        test = self.__spike_times_per_neuron
        weights_grad = compute_weights_gradient(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                self.__pre_exp_tau_s, self.__pre_exp_tau, errors)
        # Propagate errors
        # what are the dimensions of f1
        # what are the dimensions of pre_exp_tau_s
        if self.__previous_layer.trainable:
            pre_errors = propagate_errors_to_pre_spikes(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                        self.__pre_exp_tau_s, self.__pre_exp_tau, self.__weights,
                                                        errors, self.__tau_s, self.__tau)
            asdasd = 0
            # if cp.any(cp.isnan(pre_errors)):
            #     raise ValueError("NaNs in pre_errors")
        else:
            pre_errors = None

        # if cp.any(cp.isnan(weights_grad)):
        #     raise ValueError("NaNs in weights_grad")
        return weights_grad, pre_errors

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        self.__weights += delta_weights
