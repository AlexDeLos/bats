from doctest import debug
from math import inf
from operator import ne
from typing import Callable, Tuple
from typing import Optional
import cupy as cp
from cupy._core import ndarray
from cupy._creation.from_data import array
# import numpy as np

from bats.AbstractConvLayer import AbstractConvLayer
from bats.CudaKernels.Wrappers.Backpropagation.compute_weights_gradient_conv import compute_weights_gradient_conv
from bats.CudaKernels.Wrappers.Backpropagation.propagate_errors_to_pre_spikes_conv import \
    propagate_errors_to_pre_spikes_conv
from bats.CudaKernels.Wrappers.Inference import *
from bats.CudaKernels.Wrappers.Backpropagation import *
from bats.CudaKernels.Wrappers.Inference.compute_spike_times_conv import compute_spike_times_conv


class ConvLIFLayerResidual(AbstractConvLayer):
    def __init__(self, previous_layer: AbstractConvLayer, jump_layer: AbstractConvLayer, filters_shape: cp.ndarray,
                 use_padding: bool,
                 tau_s: float, theta: float,
                 delta_theta: float, # input_channels:,
                 weight_initializer: Callable[[int, int, int, int], cp.ndarray] = None, max_n_spike: int = 32,
                 **kwargs):
        prev_x, prev_y, prev_c = previous_layer._neurons_shape.get()
        prev_x_jump, prev_y_jump, prev_c_jump = jump_layer._neurons_shape.get()
        filter_x, filter_y, filter_c = filters_shape #? do I need to duplicate this?
        if use_padding:
            n_x = prev_x
            n_x_jump = prev_x_jump
            n_y = prev_y
            n_y_jump = prev_y_jump
        else:
            n_x = prev_x - filter_x + 1
            n_x_jump = prev_x_jump - filter_x + 1
            n_y = prev_y - filter_y + 1
            n_y_jump = prev_y_jump - filter_y + 1
        if n_x != n_x_jump or n_y != n_y_jump:
            raise ValueError("The dimensions of the layers do not match")
        # n_x = prev_x - filter_x + 1
        # #? why are they connected? couldn't it be whatever?
        # n_x_jump = prev_x_jump - filter_x + 1
        # n_y = prev_y - filter_y + 1
        # n_y_jump = prev_y_jump - filter_y + 1
        neurons_shape: cp.ndarray = cp.array([n_x, n_y, filter_c], dtype=cp.int32)
        #! neurons_shape just return the normal shape, not the residual shape
        neurons_shape_jump: cp.ndarray = cp.array([n_x_jump, n_y_jump, filter_c], dtype=cp.int32)
        if cp.all(neurons_shape != neurons_shape_jump):
            raise ValueError("The dimensions of the layers do not match 2")

        # how can I mix it?
        super().__init__(neurons_shape=neurons_shape, use_padding= use_padding, **kwargs)
        # super().__init__(neurons_shape=neurons_shape, **kwargs)
        self._is_residual = True
        self.fuse_function = "Append"
        self.jump_layer = jump_layer
        self.__neurons_shape_pre: cp.ndarray = cp.array([n_x, n_y, filter_c], dtype=cp.int32)
        self.__neurons_shape_jump: cp.ndarray = cp.array([n_x_jump, n_y_jump, filter_c], dtype=cp.int32)
        self.__number_of_neurons_pre = int(self.__neurons_shape_pre[0] * self.__neurons_shape_pre[1] * self.__neurons_shape_pre[2])
        self.__number_of_neurons_jump = int(self.__neurons_shape_jump[0] * self.__neurons_shape_jump[1] * self.__neurons_shape_jump[2])
        self._n_neurons = self.__number_of_neurons_jump + self.__number_of_neurons_pre

        self.__filters_shape = cp.array([filter_c, filter_x, filter_y, prev_c], dtype=cp.int32)
        self.__filters_shape_jump = cp.array([filter_c, filter_x, filter_y, prev_c_jump], dtype=cp.int32)
        self.__previous_layer: AbstractConvLayer = previous_layer
        self.__jump_layer: AbstractConvLayer = jump_layer
        self.__tau_s: cp.float32 = cp.float32(tau_s)
        self.__tau_s_jump: cp.float32 = cp.float32(tau_s)
        self.__tau: cp.float32 = cp.float32(2 * tau_s)
        self.__tau_jump: cp.float32 = cp.float32(2 * tau_s)
        self.__theta_tau: cp.float32 = cp.float32(theta / self.__tau)
        self.__theta_tau_jump: cp.float32 = cp.float32(theta / self.__tau_jump)
        self.__delta_theta_tau: cp.float32 = cp.float32(delta_theta / self.__tau)
        self.__delta_theta_tau_jump: cp.float32 = cp.float32(delta_theta / self.__tau_jump)
        if weight_initializer is None:
            self.__weights_pre: cp.ndarray = cp.zeros((filter_c, filter_x, filter_y, prev_c), dtype=cp.float32)
            self.__weights_jump: cp.ndarray = cp.zeros((filter_c, filter_x, filter_y, prev_c_jump), dtype=cp.float32)
        else:
            self.__weights_pre: cp.ndarray = weight_initializer(filter_c, filter_x, filter_y, prev_c)
            self.__weights_jump: cp.ndarray = weight_initializer(filter_c, filter_x, filter_y, prev_c_jump) 
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__n_spike_per_neuron_jump: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron_jump: Optional[cp.ndarray] = None
        self.__a: Optional[cp.ndarray] = None
        self.__a_jump: Optional[cp.ndarray] = None
        self.__x: Optional[cp.ndarray] = None
        self.__x_jump: Optional[cp.ndarray] = None
        self.__post_exp_tau: Optional[cp.ndarray] = None
        self.__post_exp_tau_jump: Optional[cp.ndarray] = None

        self.__pre_exp_tau_s: Optional[cp.ndarray] = None
        self.__pre_exp_tau_s_jump: Optional[cp.ndarray] = None
        self.__pre_exp_tau: Optional[cp.ndarray] = None
        self.__pre_exp_tau_jump: Optional[cp.ndarray] = None
        # self.__pre_spike_weights: Optional[cp.ndarray] = None
        self.__c: Optional[cp.float32] = self.__theta_tau
        self.__c_jump: Optional[cp.float32] = self.__theta_tau_jump

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        spikes, number = fuse_inputs_append(self.__spike_times_per_neuron, self.__spike_times_per_neuron_jump, self.__n_spike_per_neuron, self.__n_spike_per_neuron_jump, self.__max_n_spike)
        # No NaNs here
        return spikes, number

        return self.__spike_times_per_neuron, self.__n_spike_per_neuron

    @property
    def weights(self) -> Optional[cp.ndarray]:
        # return self.__weights_pre
        #! this needs to be updated to reflect the real weights of the layer
        # Hypothesis: we should simply add the channels of the weights
        ret = cp.append(self.__weights_pre, self.__weights_jump, axis=3)
        return ret

    @weights.setter
    def weights(self, weights: cp.ndarray) -> None:
        self.__weights_pre = cp.array(weights, dtype=cp.float32)

    def reset(self) -> None:
        self.__n_spike_per_neuron = None
        self.__spike_times_per_neuron = None
        self.__pre_exp_tau_s = None
        self.__pre_exp_tau = None
        self.__pre_spike_weights = None
        self.__a = None
        self.__x = None
        self.__post_exp_tau = None

    def forward_pre(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__previous_layer.spike_trains

        self.__pre_exp_tau_s, self.__pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)

        # Sort spikes for inference
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(pre_spike_per_neuron,
                                                                                    pre_n_spike_per_neuron)
        if sorted_indices.size == 0:  # No input spike in the batch
            batch_size = pre_spike_per_neuron.shape[0]
            shape = (batch_size, self.__number_of_neurons_pre, self.__max_n_spike)
            self.__n_spike_per_neuron = cp.zeros((batch_size, self.__number_of_neurons_pre), dtype=cp.int32) #! errors here
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
            self.__post_exp_tau = compute_spike_times_conv(sorted_spike_indices, sorted_spike_times,
                                                           sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                           self.weights, self.__c, self.__delta_theta_tau,
                                                           self.__tau, cp.float32(max_simulation), self.__max_n_spike,
                                                           self.__previous_layer.neurons_shape, self.__neurons_shape_pre,
                                                           self.__filters_shape)
            test = self.__n_spike_per_neuron
            test2 = self.__spike_times_per_neuron
            test3 = self.__post_exp_tau
            test4 = self.__a
            test5 = self.__x
            w = self.__weights_pre
            tyu = 0
    
    def forward_jump(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__jump_layer.spike_trains

        self.__pre_exp_tau_s_jump, self.__pre_exp_tau_jump = compute_pre_exps(pre_spike_per_neuron, self.__tau_s_jump, self.__tau_jump)
        # Sort spikes for inference
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(pre_spike_per_neuron,
                                                                                    pre_n_spike_per_neuron)
        if sorted_indices.size == 0:  # No input spike in the batch
            batch_size = pre_spike_per_neuron.shape[0]
            shape = (batch_size, self.__number_of_neurons_jump, self.__max_n_spike) #! shape will cause problems
            self.__n_spike_per_neuron_jump = cp.zeros((batch_size, self.__number_of_neurons_jump), dtype=cp.int32)
            self.__spike_times_per_neuron_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__post_exp_tau_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__a_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__x_jump = cp.full(shape, cp.inf, dtype=cp.float32)
        else:
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)
            sorted_spike_times[sorted_indices == -1] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_s_jump, new_shape), sorted_indices,
                                                      axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_jump, new_shape), sorted_indices, axis=1)

            self.__n_spike_per_neuron_jump, self.__a_jump, self.__x_jump, self.__spike_times_per_neuron_jump, \
            self.__post_exp_tau_jump = compute_spike_times_conv(sorted_spike_indices, sorted_spike_times,
                                                           sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                           self.__weights_jump, self.__c_jump, self.__delta_theta_tau_jump,
                                                           self.__tau_jump, cp.float32(max_simulation), self.__max_n_spike,
                                                           self.__jump_layer.neurons_shape, self.__neurons_shape_jump,
                                                           self.__filters_shape_jump)
            # No NaNs here
            test = self.__n_spike_per_neuron_jump
            test2 = self.__spike_times_per_neuron_jump
            test3 = self.__post_exp_tau_jump
            test4 = self.__a_jump
            test5 = self.__x_jump
            w = self.__weights_jump
            tyu = 0

    def forward(self, max_simulation: float, training: bool = False) -> None:
        self.forward_pre(max_simulation, training)
        self.forward_jump(max_simulation, training)

    def backward_pre(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        #shape of errors is (batch_size, n_neurons, ?(3)) -> yes
        pre_spike_per_neuron, _ = self.__previous_layer.spike_trains
        # shape of pre_spike_per_neuron is (batch_size, n_neurons_pre_pre, max_n_spike(3)) -> Yes
        # __spike_times_per_neuron.shape => (batch_size, n_neurons_current_pre, max_n_spike) -> Yes
        inputx = self.__x
        input__post_exp_tau = self.__post_exp_tau
        propagate_recurrent_errors(self.__x, self.__post_exp_tau, errors, self.__delta_theta_tau)
        f1, f2 = compute_factors(self.__spike_times_per_neuron, self.__a, self.__c, self.__x,
                                 self.__post_exp_tau, self.__tau)
        # the shape for everything here is (batch_size, n_neurons_current_pre, max_n_spike) -> make sure they fit with the weights
        #! check the shapes of the inputs

        # weights_grad.shape => (batch_size, (self.__filters_shape_jump.shape))
        # self.__filters_shape_jump.shape => (filter_c, filter_x, filter_y, prev_c)
        input1 = self.__pre_exp_tau_s_jump
        input2 = self.__pre_exp_tau_jump
        input3 = self.__weights_jump# -> why is the shape 40? input3.shape => (40, 5, 5, 15)
        input4 = self.__neurons_shape_jump
        test1 = cp.any(cp.isnan(input1))
        test2 = cp.any(cp.isnan(input2))
        test3 = cp.any(cp.isnan(input3))
        test4 = cp.any(cp.isnan(input4))
        weights_grad = compute_weights_gradient_conv(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                     self.__pre_exp_tau_s, self.__pre_exp_tau,
                                                     self.__previous_layer.neurons_shape,
                                                     self.__neurons_shape_pre,
                                                     self.__filters_shape,
                                                     errors)

        # Propagate errors
        if self.__previous_layer.trainable:
            pre_errors = propagate_errors_to_pre_spikes_conv(f1, f2, self.__spike_times_per_neuron,
                                                             pre_spike_per_neuron,
                                                             self.__pre_exp_tau_s, self.__pre_exp_tau, self.__weights_pre,
                                                             errors, self.__tau_s, self.__tau,
                                                             self.__previous_layer.neurons_shape,
                                                             self.__neurons_shape_pre, #! what is this in the jump layer?
                                                             self.__filters_shape)
        else:
            pre_errors = None

        return weights_grad, pre_errors
    
    def backward_jump(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # SAVE ERROR FOR DEBUG PURPOSES
        #! should errors be infinities?
        #* that is not the problem
        
        debug_err = errors.copy()

        # try to overwrite the infs in the errors
        #! OK, this would fix it, but why are there infs in the first place?

        # Compute gradient
        #shape of errors is (batch_size, n_neurons, ?(3))
        pre_spike_per_neuron, spikes = self.__jump_layer.spike_trains
        
        # shape of pre_spike_per_neuron is (batch_size, n_neurons_jump, max_n_spike)
        #! what  about the shape of pre_spike_per_neuron
        #! do self.__spike_times_per_neuron_jump have the proper shape?
        # __spike_times_per_neuron_jump.shape => (batch_size, n_neurons_jump, max_n_spike)
        #! this adds nans to the errors
        inputx = self.__x
        input__post_exp_tau = self.__post_exp_tau
        #! the problem is not the errors, replaced the errors_pre and still got nans
        #! now trying to see if the problem is with the variables of the spikes
        #! self.__x_jump, self.__post_exp_tau_jump seem to be the problem
        propagate_recurrent_errors(self.__x_jump, self.__post_exp_tau_jump, errors, self.__delta_theta_tau_jump)
        # propagate recurrent errors changes the errors variable, not the shape
        f1, f2 = compute_factors(self.__spike_times_per_neuron_jump, self.__a_jump, self.__c_jump, self.__x_jump,
                                 self.__post_exp_tau_jump, self.__tau_jump)
        # the shape for everything here is (batch_size, n_neurons, max_n_spike) -> make sure they fit with the weights
        #! check the shapes of the inputs

        # weights_grad.shape => (batch_size, (self.__filters_shape_jump.shape))
        # self.__filters_shape_jump.shape => (filter_c, filter_x, filter_y, prev_c)
        input1 = self.__pre_exp_tau_s_jump
        input2 = self.__pre_exp_tau_jump
        input3 = self.__weights_jump# -> why is the shape 40? input3.shape => (40, 5, 5, 15)
        input4 = self.__neurons_shape_jump
        test1 = cp.any(cp.isnan(input1))
        test2 = cp.any(cp.isnan(input2))
        test3 = cp.any(cp.isnan(input3))
        test4 = cp.any(cp.isnan(input4))

        weights_grad = compute_weights_gradient_conv(f1, f2, errors, pre_spike_per_neuron,
                                                     self.__pre_exp_tau_s_jump, self.__pre_exp_tau_jump,
                                                     self.__jump_layer.neurons_shape,
                                                     self.__neurons_shape_jump,
                                                     self.__filters_shape_jump,
                                                     errors)
        # weights_grad = cp.nan_to_num(weights_grad, nan=0.0)

        # Propagate errors
        #! all nans on the last batch, why?
        #? does this happen repeatedly?
        if self.__jump_layer.trainable:
            # pre_errors.shape => (batch_size, n_neurons_jump, max_n_spike_jump)
            #! after a few iterations there are nans in the pre_errors
            #! The problem comes when the errors that come in contain infs
            pre_errors = propagate_errors_to_pre_spikes_conv(f1, f2, self.__spike_times_per_neuron_jump,
                                                             pre_spike_per_neuron,
                                                             self.__pre_exp_tau_s_jump, self.__pre_exp_tau_jump, self.__weights_jump,
                                                             errors, self.__tau_s_jump, self.__tau_jump,
                                                             self.__jump_layer.neurons_shape,
                                                             self.__neurons_shape_jump, #! this need to be doubled?
                                                             self.__filters_shape_jump)
        else:
            pre_errors = None
        return weights_grad, pre_errors
    
    def backward(self, errors: cp.array) -> Tuple:
        #TODO: split the errors into pre and jump
        split_index = self.__number_of_neurons_pre
        split_index_jump = self.__number_of_neurons_jump
        errors_pre, errors_jump = cp.split(errors, [int(split_index)], axis=1)
        if split_index != errors_pre.shape[1]:
            raise ValueError("The split index is not correct")
        if split_index_jump != errors_jump.shape[1]:
            raise ValueError("The split index is not correct")
        # if the error size is to big it gives nans 
        weights_grad_pre, pre_errors_pre = self.backward_pre(errors_pre)
        #! when i put errors I get a similar type nans
        weights_grad_jump, pre_errors_jump = self.backward_jump(errors_jump)

        #? is the problem with the error input?
        #* if I use the errors_pre here I get NO nans

        #problem with the input?
        #! NaNs show up here
        testing_break = "s"
        return (weights_grad_pre, weights_grad_jump), (pre_errors_pre,pre_errors_jump)
        

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        self.__weights_pre += delta_weights[0]
        self.__weights_jump += delta_weights[1]


def fuse_inputs_append(residual_input, jump_input, count_residual, count_jump, max_n_spike, delay = None) -> Tuple[cp.ndarray, cp.ndarray]:
    batch_size_res, n_of_neurons_res, max_n_spike_res = residual_input.shape
    batch_size_jump, n_of_neurons_jump, max_n_spike_jump = jump_input.shape
    # I need to make sure I append the right way with the channel dimension

    result_count =cp.append(count_residual, count_jump, axis=1)
    # this changes the effect
    # result_count = count_residual
    # result_count = cp.zeros((residual_input.shape))
    result_spikes = cp.append(residual_input, jump_input, axis=1)
    if cp.any(result_count > max_n_spike):
        raise ValueError("The count of spikes is greater than the max number of spikes")
    # result_count = count_residual
    # result_spikes = residual_input
    return result_spikes, result_count