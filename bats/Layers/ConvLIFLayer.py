from multiprocessing import Pool
from typing import Callable, Tuple
from typing import Optional
import cupy as cp
import numpy as np

from bats.AbstractConvLayer import AbstractConvLayer
from bats.CudaKernels.Wrappers.Backpropagation.compute_weights_gradient_conv import compute_weights_gradient_conv
from bats.CudaKernels.Wrappers.Backpropagation.propagate_errors_to_pre_spikes_conv import \
    propagate_errors_to_pre_spikes_conv
from bats.CudaKernels.Wrappers.Inference import *
from bats.CudaKernels.Wrappers.Backpropagation import *
from bats.CudaKernels.Wrappers.Inference.compute_spike_times_conv import compute_spike_times_conv
from bats.Layers import PoolingLayer
from bats.Utils.utils import add_padding, trimed_errors


class ConvLIFLayer(AbstractConvLayer):
    def __init__(self, previous_layer: AbstractConvLayer, filters_shape: np.ndarray, use_padding: bool,
                 tau_s: float, theta: float,
                 delta_theta: float,
                 filter_from_next: cp.ndarray = None,
                 weight_initializer: Callable[[int, int, int, int], cp.ndarray] = None, max_n_spike: int = 32,
                 **kwargs):
        prev_x, prev_y, prev_c = previous_layer._neurons_shape.get()
        filter_x, filter_y, filter_c = filters_shape
        padding= [filter_x-1, filter_y -1]
        self.__filter_from_next = filter_from_next
        if use_padding:
            prev_x += padding[0]
            prev_y += padding[1]
        self.__pre_shape = (prev_x, prev_y, prev_c)
        n_x = prev_x - filter_x + 1 # why this equation? -> this is the reduction of dimensions because of the filter
        n_y = prev_y - filter_y + 1
        neurons_shape: cp.ndarray = cp.array([n_x, n_y, filter_c], dtype=cp.int32)

        super().__init__(neurons_shape=neurons_shape, use_padding = use_padding,padding= [filter_x-1, filter_y -1], **kwargs)

        self.__filters_shape = cp.array([filter_c, filter_x, filter_y, prev_c], dtype=cp.int32)
        self.__previous_layer: AbstractConvLayer = previous_layer
        self.__tau_s: cp.float32 = cp.float32(tau_s)
        self.__tau: cp.float32 = cp.float32(2 * tau_s)
        self.__theta_tau: cp.float32 = cp.float32(theta / self.__tau)
        self.__delta_theta_tau: cp.float32 = cp.float32(delta_theta / self.__tau)
        if weight_initializer is None:
            self.__weights: cp.ndarray = cp.zeros((filter_c, filter_x, filter_y, prev_c), dtype=cp.float32)
        else:
            self.__weights: cp.ndarray = weight_initializer(filter_c, filter_x, filter_y, prev_c)
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__a: Optional[cp.ndarray] = None
        self.__x: Optional[cp.ndarray] = None
        self.__post_exp_tau: Optional[cp.ndarray] = None

        self.__pre_exp_tau_s: Optional[cp.ndarray] = None
        self.__padded_pre_exp_tau_s: Optional[cp.ndarray] = None
        self.__pre_exp_tau: Optional[cp.ndarray] = None
        self.__padded_pre_exp_tau: Optional[cp.ndarray] = None
        self.__pre_spike_weights: Optional[cp.ndarray] = None
        self.__c: Optional[cp.float32] = self.__theta_tau

        self.__pre_spike_per_neuron: Optional[cp.ndarray] = None
        self.__pre_n_spike_per_neuron: Optional[cp.ndarray] = None


    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
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
        self.__pre_spike_weights = None
        self.__a = None
        self.__x = None
        self.__post_exp_tau = None

    def forward(self, max_simulation: float, training: bool = False) -> None:
        if not self._use_padding:
            self.forward_no_pad(max_simulation, training)
            return        
        pre_spike_per_neuron_pre_pad, pre_n_spike_per_neuron_pre_pad = self.__previous_layer.spike_trains
        # print(self.name)
        # print(pre_n_spike_per_neuron_pre_pad)
        # print(cp.where(pre_n_spike_per_neuron_pre_pad!=0))

        # should have the size of the previous layer
        # self.__pre_exp_tau_s, self.__pre_exp_tau = compute_pre_exps(pre_spike_per_neuron_pre_pad, self.__tau_s, self.__tau) 
        
        # how are the spikes used? and how do I add padding?
        #? what is I don't use padding in the forward pass?
        self.__pre_spike_per_neuron, self.__pre_n_spike_per_neuron = add_padding(pre_spike_per_neuron_pre_pad, pre_n_spike_per_neuron_pre_pad, self.__pre_shape, self._padding)
        pre_spike_per_neuron = self.__pre_spike_per_neuron
        pre_n_spike_per_neuron = self.__pre_n_spike_per_neuron
        self.__padded_pre_exp_tau_s, self.__padded_pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)
        padded_pre_exp_tau_s = self.__padded_pre_exp_tau_s
        padded_pre_exp_tau = self.__padded_pre_exp_tau
        new_shape_previous = self.__pre_shape

        # Sort spikes for inference
        #? what if I add padding to the indeces? increase theyr number to what they should be?
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
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)
            sorted_spike_times[sorted_indices == -1] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(padded_pre_exp_tau_s, new_shape), sorted_indices,
                                                      axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(padded_pre_exp_tau, new_shape), sorted_indices, axis=1)
            
            # print(self.name)
            # print(cp.where(sorted_spike_indices != 0))
            # print(sorted_spike_times)
            # print(new_shape_previous)
            #? this line works on computer but not on the cluster
            new_shape_previous = cp.array(new_shape_previous)
            #? and if you do it like this it works on the cluster
            new_shape_previous = self.__previous_layer.neurons_shape #-> I dont like this
            self.__n_spike_per_neuron, self.__a, self.__x, self.__spike_times_per_neuron, \
            self.__post_exp_tau = compute_spike_times_conv(sorted_spike_indices, sorted_spike_times,
                                                           sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                           self.weights, self.__c, self.__delta_theta_tau,
                                                           self.__tau, cp.float32(max_simulation), self.__max_n_spike,
                                                           new_shape_previous, self.neurons_shape,
                                                           self.__filters_shape)
            # # #? what does the X represent?
            # count = self.__n_spike_per_neuron
            # print(self.name)
            # print(cp.where(count!=0)[0].shape)
            ewrwe = 0

    def forward_no_pad(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__previous_layer.spike_trains

        self.__pre_exp_tau_s, self.__pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)

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
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)
            sorted_spike_times[sorted_indices == -1] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_s, new_shape), sorted_indices,
                                                      axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(self.__pre_exp_tau, new_shape), sorted_indices, axis=1)

            self.__n_spike_per_neuron, self.__a, self.__x, self.__spike_times_per_neuron, \
            self.__post_exp_tau = compute_spike_times_conv(sorted_spike_indices, sorted_spike_times, # some other size
                                                           sorted_pre_exp_tau_s, sorted_pre_exp_tau,# some other size
                                                           self.weights, self.__c, self.__delta_theta_tau,
                                                           self.__tau, cp.float32(max_simulation), self.__max_n_spike,
                                                           self.__previous_layer.neurons_shape, self.neurons_shape,
                                                           self.__filters_shape)
            asd = ''


    def backward(self, errors_in: cp.array, from_res = False) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        if cp.any(cp.isnan(errors_in)):
            raise ValueError("NaNs in errors")
        if not self._use_padding:
            return self.backward_no_pad(errors_in)
        pre_spike_per_neuron_real, _ = self.__previous_layer.spike_trains
        
        #? what is I don't use padding in the backward pass?
        
        pre_spike_per_neuron = self.__pre_spike_per_neuron #? this is the padded, what if I don't feed it the padded?
        new_shape_previous = (self.__previous_layer.neurons_shape[0]+ self._padding[0], self.__previous_layer.neurons_shape[1] + self._padding[1], self.__previous_layer.neurons_shape[2])
        # pre_spike_per_neuron = pre_spike_per_neuron_real #?


        new_shape_previous = cp.array(new_shape_previous, dtype=cp.int32)
        # new_shape_previous = self.__previous_layer.neurons_shape
        padded_pre_exp_tau_s = self.__padded_pre_exp_tau_s
        padded_pre_exp_tau = self.__padded_pre_exp_tau
        # padded_pre_exp_tau_s = self.__pre_exp_tau_s
        # padded_pre_exp_tau = self.__pre_exp_tau
        new_x = self.__x
        new_post_exp_tau = self.__post_exp_tau
        if new_x.shape != errors_in.shape:
            print("Error in the shapes")
            raise RuntimeError("This should not happen")
        else:
            errors = errors_in
        
        new_spike_times_per_neuron = self.__spike_times_per_neuron #? what is this?
        errors_debug = errors_in.copy()
        
        propagate_recurrent_errors(new_x, new_post_exp_tau, errors, self.__delta_theta_tau)#! they all have the shape of the current layer
        f1, f2 = compute_factors(new_spike_times_per_neuron, self.__a, self.__c, new_x,
                                 new_post_exp_tau, self.__tau)
        weights_grad = compute_weights_gradient_conv(f1, f2, self.__spike_times_per_neuron, # this has the shape of the current layer
                                                     pre_spike_per_neuron,# this has the shape of the previous layer
                                                     padded_pre_exp_tau_s, padded_pre_exp_tau, # these 2 also have the shape of the previous layer
                                                     #! BUT IF I TRY TO run it as is now I get the shape of the current layer
                                                    #  new_shape_previous,
                                                     self.__previous_layer.neurons_shape,
                                                     self.neurons_shape,
                                                     self.__filters_shape,
                                                     errors)# this has the shape of the current layer

        # Propagate errors
        if self.__previous_layer.trainable:
            # the error shape comes from: 
            pre_errors = propagate_errors_to_pre_spikes_conv(f1, f2, self.__spike_times_per_neuron, # this has the shape of the current layer
                                                             pre_spike_per_neuron, # this has the shape of the previous layer
                                                             padded_pre_exp_tau_s, padded_pre_exp_tau, # these 2 also have the shape of the previous layer
                                                             self.__weights,
                                                             errors, # this has the shape of the current layer
                                                             self.__tau_s, self.__tau,
                                                             new_shape_previous,
                                                            #  self.__previous_layer.neurons_shape, #!  these 2 shapes are out of date with the padding
                                                             self.neurons_shape,
                                                             self.__filters_shape)
            testsd = 0
        else:
            pre_errors = None

        stop_for_errors = 0
        #? should I do some reshaping of the pre_errors?
        pre_layer = self.__previous_layer
        #! this is a bad fix 
        if pre_layer.name.__contains__("Pooling") or pre_errors is None:
            pass
        else:
            pass
        #? if using this code where we don't use padding in the backward pass, we need not 
            pre_errors = trimed_errors(pre_errors, self.__filters_shape, self.__filters_shape[3])

        return weights_grad, pre_errors
    
    def backward_no_pad(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        pre_spike_per_neuron, _ = self.__previous_layer.spike_trains
        propagate_recurrent_errors(self.__x, self.__post_exp_tau, errors, self.__delta_theta_tau)
        f1, f2 = compute_factors(self.__spike_times_per_neuron, self.__a, self.__c, self.__x,
                                 self.__post_exp_tau, self.__tau)
        weights_grad = compute_weights_gradient_conv(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                     self.__pre_exp_tau_s, self.__pre_exp_tau,
                                                     self.__previous_layer.neurons_shape,
                                                     self.neurons_shape,
                                                     self.__filters_shape,
                                                     errors)

        # Propagate errors
        if self.__previous_layer.trainable:
            pre_errors = propagate_errors_to_pre_spikes_conv(f1, f2, self.__spike_times_per_neuron,
                                                             pre_spike_per_neuron,
                                                             self.__pre_exp_tau_s, self.__pre_exp_tau, self.__weights,
                                                             errors, self.__tau_s, self.__tau,
                                                             self.__previous_layer.neurons_shape,
                                                             self.neurons_shape,
                                                             self.__filters_shape)
        else:
            pre_errors = None

        #? maybe trim the errors here

        return weights_grad, pre_errors

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        self.__weights += delta_weights