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
from bats.Utils.utils import add_padding, trimed_errors, aped_on_channel_dim, split_on_channel_dim

class ConvLIFLayerResidual_2(AbstractConvLayer):
    def __init__(self, previous_layer: AbstractConvLayer, jump_layer: AbstractConvLayer, filters_shape: cp.ndarray,
                 use_padding: bool,
                 tau_s: float, theta: float,
                 delta_theta: float, # input_channels:,
                 filter_from_next: cp.ndarray = None,
                 weight_initializer: Callable[[int, int, int, int], cp.ndarray] = None, max_n_spike: int = 32,
                 **kwargs):
        prev_x, prev_y, prev_c = previous_layer._neurons_shape.get()
        prev_x_jump, prev_y_jump, prev_c_jump = jump_layer._neurons_shape.get()
        if prev_x != prev_x_jump or prev_y != prev_y_jump:
            raise ValueError("The input dimensions of the previous layers are not the same")
        filter_x, filter_y, filter_c = filters_shape
        padding= [filter_x-1, filter_y -1]
        self.__filter_from_next = filter_from_next
        if use_padding:
            prev_x += padding[0]
            prev_y += padding[1]
        self.__pre_shape = (prev_x, prev_y, prev_c)
        self.__jump_shape = (prev_x_jump, prev_y_jump, prev_c_jump)
        n_x = prev_x - filter_x + 1 # why this equation? -> this is the reduction of dimensions because of the filter
        n_y = prev_y - filter_y + 1
        neurons_shape: cp.ndarray = cp.array([n_x, n_y, filter_c], dtype=cp.int32)

        super().__init__(neurons_shape=neurons_shape, use_padding = use_padding,padding= [filter_x-1, filter_y -1], **kwargs)
        
        # super().__init__(neurons_shape=neurons_shape, **kwargs)
        self._is_residual = True
        self.fuse_function = "Append"
        self.jump_layer = jump_layer
        self.__neurons_shape_pre: cp.ndarray = cp.array([n_x, n_y, filter_c], dtype=cp.int32)
        self.__neurons_shape_jump: cp.ndarray = cp.array([n_x, n_y, filter_c], dtype=cp.int32)
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
        self.__padded_pre_exp_tau_s: Optional[cp.ndarray] = None
        self.__pre_exp_tau: Optional[cp.ndarray] = None
        self.__padded_pre_exp_tau: Optional[cp.ndarray] = None

        self.__pre_exp_tau_s_jump: Optional[cp.ndarray] = None
        self.__padded_pre_exp_tau_s_jump: Optional[cp.ndarray] = None
        self.__pre_exp_tau_jump: Optional[cp.ndarray] = None
        self.__padded_pre_exp_tau_jump: Optional[cp.ndarray] = None
        # self.__pre_spike_weights: Optional[cp.ndarray] = None
        self.__c: Optional[cp.float32] = self.__theta_tau
        self.__c_jump: Optional[cp.float32] = self.__theta_tau_jump


        self.__pre_spike_per_neuron: Optional[cp.ndarray] = None
        self.__pre_n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__jump_spike_per_neuron: Optional[cp.ndarray] = None
        self.__jump_n_spike_per_neuron: Optional[cp.ndarray] = None

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        spikes, number = aped_on_channel_dim(self.__spike_times_per_neuron, self.__n_spike_per_neuron,
                                             self.__spike_times_per_neuron_jump, self.__n_spike_per_neuron_jump,
                                            #  self.__spike_times_per_neuron, self.__n_spike_per_neuron,
                                            self.neurons_shape, self.neurons_shape)
        # No NaNs here
        return spikes, number

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
        self.__n_spike_per_neuron_jump = None
        self.__spike_times_per_neuron = None
        self.__spike_times_per_neuron_jump = None
        self.__pre_exp_tau_s = None
        self.__pre_exp_tau_s_jump = None
        
        self.__pre_exp_tau = None
        self.__pre_exp_tau_jump = None
        self.__padded_pre_exp_tau = None
        self.__padded_pre_exp_tau_jump = None

        self.__pre_spike_weights = None

        self.__a = None
        self.__a_jump = None
        self.__x = None
        self.__x_jump = None
        self.__post_exp_tau = None
        self.__post_exp_tau_jump = None

    def forward_pre(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__previous_layer.spike_trains
        
        # how are the spikes used? and how do I add padding?
        if self._use_padding: #! using this causes random nans for some reason
        #? what is I don't use padding in the forward pass?
            self.__pre_spike_per_neuron, self.__pre_n_spike_per_neuron = add_padding(pre_spike_per_neuron, pre_n_spike_per_neuron,
                                                                       self.__pre_shape, self._padding)
            pre_spike_per_neuron = self.__pre_spike_per_neuron
            pre_n_spike_per_neuron = self.__pre_n_spike_per_neuron
            self.__padded_pre_exp_tau_s, self.__padded_pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)
            padded_pre_exp_tau_s = self.__padded_pre_exp_tau_s
            padded_pre_exp_tau = self.__padded_pre_exp_tau
            new_shape_previous = (self.__previous_layer.neurons_shape[0]+ self._padding[0], self.__previous_layer.neurons_shape[1] + self._padding[1], self.__previous_layer.neurons_shape[2])
            new_shape_previous = cp.array(new_shape_previous, dtype=cp.int32)
        else:
            self.__pre_spike_per_neuron, self.__pre_n_spike_per_neuron = pre_spike_per_neuron, pre_n_spike_per_neuron
            self.__pre_exp_tau_s, self.__pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)
            padded_pre_exp_tau_s = self.__pre_exp_tau_s
            padded_pre_exp_tau = self.__pre_exp_tau
            new_shape_previous = self.__previous_layer.neurons_shape

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

            self.__n_spike_per_neuron, self.__a, self.__x, self.__spike_times_per_neuron, \
            self.__post_exp_tau = compute_spike_times_conv(sorted_spike_indices, sorted_spike_times,
                                                           sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                           self.weights, self.__c, self.__delta_theta_tau,
                                                           self.__tau, cp.float32(max_simulation), self.__max_n_spike,
                                                           new_shape_previous, self.neurons_shape,
                                                           self.__filters_shape)
    
    def forward_jump(self, max_simulation: float, training: bool = False) -> None:
        #TODO: change the names of the self.__ variables to the jump version
        jump_spike_per_neuron, jump_n_spike_per_neuron = self.__jump_layer.spike_trains
        
        # how are the spikes used? and how do I add padding?
        if self._use_padding: #! using this causes random nans for some reason
        #? what is I don't use padding in the forward pass?
            self.__jump_spike_per_neuron, self.__jump_n_spike_per_neuron = add_padding(jump_spike_per_neuron, jump_n_spike_per_neuron,
                                                                       self.__jump_shape, self._padding)
            
            jump_spike_per_neuron = self.__jump_spike_per_neuron
            jump_n_spike_per_neuron = self.__jump_n_spike_per_neuron
            self.__padded_pre_exp_tau_s_jump, self.__padded_pre_exp_tau_jump = compute_pre_exps(jump_spike_per_neuron, self.__tau_s, self.__tau)
            padded_pre_exp_tau_s_jump = self.__padded_pre_exp_tau_s_jump
            padded_pre_exp_tau_jump = self.__padded_pre_exp_tau_jump
            new_shape_previous = (self.__jump_layer.neurons_shape[0]+ self._padding[0], self.__jump_layer.neurons_shape[1] + self._padding[1], self.__jump_layer.neurons_shape[2])
            new_shape_previous = cp.array(new_shape_previous, dtype=cp.int32)
        else:
            self.__jump_spike_per_neuron, self.__jump_n_spike_per_neuron = jump_spike_per_neuron, jump_n_spike_per_neuron
            self.__pre_exp_tau_s_jump, self.__pre_exp_tau_jump = compute_pre_exps(jump_spike_per_neuron, self.__tau_s_jump, self.__tau_jump)
            padded_pre_exp_tau_s_jump = self.__pre_exp_tau_s_jump
            padded_pre_exp_tau_jump = self.__pre_exp_tau_jump
            new_shape_previous = self.__jump_layer.neurons_shape

        # Sort spikes for inference
        #? what if I add padding to the indeces? increase theyr number to what they should be?
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(jump_spike_per_neuron,
                                                                                    jump_n_spike_per_neuron)
        if sorted_indices.size == 0:  # No input spike in the batch
            batch_size = jump_spike_per_neuron.shape[0]
            shape = (batch_size, self.n_neurons, self.__max_n_spike)
            self.__n_spike_per_neuron_jump = cp.zeros((batch_size, self.n_neurons), dtype=cp.int32)
            self.__spike_times_per_neuron_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__post_exp_tau_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__a_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__x_jump = cp.full(shape, cp.inf, dtype=cp.float32)
        else:
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // jump_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)
            sorted_spike_times[sorted_indices == -1] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(padded_pre_exp_tau_s_jump, new_shape), sorted_indices,
                                                      axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(padded_pre_exp_tau_jump, new_shape), sorted_indices, axis=1)

            self.__n_spike_per_neuron_jump, self.__a_jump, self.__x_jump, self.__spike_times_per_neuron_jump, \
            self.__post_exp_tau_jump = compute_spike_times_conv(sorted_spike_indices, sorted_spike_times,
                                                           sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                           self.__weights_jump, self.__c_jump, self.__delta_theta_tau_jump,
                                                           self.__tau_jump, cp.float32(max_simulation), self.__max_n_spike,
                                                           new_shape_previous, self.neurons_shape,
                                                           self.__filters_shape_jump)

    def forward(self, max_simulation: float, training: bool = False) -> None:
        self.forward_pre(max_simulation, training)
        self.forward_jump(max_simulation, training)
        te = ''

    def backward_pre(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        if cp.any(cp.isnan(errors)):
            raise ValueError("NaNs in errors")
        pre_spike_per_neuron, _ = self.__previous_layer.spike_trains
        if self._use_padding: #-> adding this alone seems to have no effect on the loss of the model or anything else
            #? what is I don't use padding in the backward pass?
            
            pre_spike_per_neuron = self.__pre_spike_per_neuron
            # pre_n_spike_per_neuron = self.__pre_n_spike_per_neuron
            new_shape_previous = (self.__previous_layer.neurons_shape[0]+ self._padding[0], self.__previous_layer.neurons_shape[1] + self._padding[1], self.__previous_layer.neurons_shape[2])
            new_shape_previous = cp.array(new_shape_previous, dtype=cp.int32)
            padded_pre_exp_tau_s = self.__padded_pre_exp_tau_s
            padded_pre_exp_tau = self.__padded_pre_exp_tau
        else:
            padded_pre_exp_tau_s = self.__pre_exp_tau_s
            padded_pre_exp_tau = self.__pre_exp_tau
            new_shape_previous = self.__previous_layer.neurons_shape

        new_x = self.__x
        new_post_exp_tau = self.__post_exp_tau
        if new_x.shape != errors.shape:
            # Errors should have already been trimmed
            raise ValueError(f"Shapes of new_x and errors do not match: {new_x.shape} != {errors.shape}")
        
        new_spike_times_per_neuron = self.__spike_times_per_neuron
        errors_debug = errors.copy()
        propagate_recurrent_errors(new_x, new_post_exp_tau, errors, self.__delta_theta_tau)#! they all have the shape of the current layer
        f1, f2 = compute_factors(new_spike_times_per_neuron, self.__a, self.__c, new_x,
                                 new_post_exp_tau, self.__tau)
        weights_grad = compute_weights_gradient_conv(f1, f2, self.__spike_times_per_neuron, # this has the shape of the current layer
                                                     pre_spike_per_neuron,# this has the shape of the previous layer
                                                     padded_pre_exp_tau_s, padded_pre_exp_tau, # these 2 also have the shape of the previous layer
                                                     #! BUT IF i TRY TO run it as is now I get the shape of the current layer
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
                                                             self.__weights_pre,
                                                             errors, # this has the shape of the current layer
                                                             self.__tau_s, self.__tau,
                                                            #  new_shape_previous,
                                                             self.__previous_layer.neurons_shape,
                                                             self.neurons_shape,
                                                             self.__filters_shape)
            testsd = 0
        else:
            pre_errors = None

        stop_for_errors = 0
        #? should I do some reshaping of the pre_errors?
        return weights_grad, pre_errors
    
    def backward_jump(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        if cp.any(cp.isnan(errors)):
            raise ValueError("NaNs in errors")
        jump_spike_per_neuron, jump_n_spike_per_neuron = self.__jump_layer.spike_trains
        if self._use_padding: #-> adding this alone seems to have no effect on the loss of the model or anything else
            #? what is I don't use padding in the backward pass?
            
            jump_spike_per_neuron = self.__jump_spike_per_neuron
            jump_n_spike_per_neuron = self.__jump_n_spike_per_neuron
            new_shape_previous = (self.__jump_layer.neurons_shape[0]+ self._padding[0], self.__jump_layer.neurons_shape[1] + self._padding[1], self.__jump_layer.neurons_shape[2])
            new_shape_previous = cp.array(new_shape_previous, dtype=cp.int32)
            padded_pre_exp_tau_s = self.__padded_pre_exp_tau_s_jump
            padded_pre_exp_tau = self.__padded_pre_exp_tau_s_jump
        else:
            padded_pre_exp_tau_s = self.__pre_exp_tau_s_jump
            padded_pre_exp_tau = self.__pre_exp_tau_jump
            new_shape_previous = self.__jump_layer.neurons_shape

        new_x = self.__x_jump
        new_post_exp_tau = self.__post_exp_tau_jump
        if new_x.shape != errors.shape:
            # Errors should have already been trimmed
            raise ValueError(f"Shapes of new_x and errors do not match: {new_x.shape} != {errors.shape}, errors should have already been trimmed")
        
        new_spike_times_per_neuron = self.__spike_times_per_neuron_jump
        errors_debug = errors.copy()
        propagate_recurrent_errors(new_x, new_post_exp_tau, errors, self.__delta_theta_tau_jump)#! they all have the shape of the current layer
        f1, f2 = compute_factors(new_spike_times_per_neuron, self.__a_jump, self.__c_jump, new_x,
                                 new_post_exp_tau, self.__tau_jump)
        weights_grad = compute_weights_gradient_conv(f1, f2, self.__spike_times_per_neuron_jump, # this has the shape of the current layer
                                                     jump_spike_per_neuron,# this has the shape of the previous layer
                                                     padded_pre_exp_tau_s, padded_pre_exp_tau, # these 2 also have the shape of the previous layer
                                                     #! BUT IF i TRY TO run it as is now I get the shape of the current layer
                                                    #  new_shape_previous,
                                                     self.__jump_layer.neurons_shape,
                                                     self.neurons_shape, #? does this need to be changed?
                                                     self.__filters_shape_jump,
                                                     errors)# this has the shape of the current layer

        # Propagate errors
        if self.__previous_layer.trainable:
            # the error shape comes from: 
            pre_errors = propagate_errors_to_pre_spikes_conv(f1, f2, self.__spike_times_per_neuron_jump, # this has the shape of the current layer
                                                             jump_spike_per_neuron, # this has the shape of the previous layer
                                                             padded_pre_exp_tau_s, padded_pre_exp_tau, # these 2 also have the shape of the previous layer
                                                             self.__weights_jump,
                                                             errors, # this has the shape of the current layer
                                                             self.__tau_s_jump, self.__tau_jump,
                                                            #  new_shape_previous,
                                                             self.__jump_layer.neurons_shape,
                                                             self.neurons_shape,
                                                             self.__filters_shape)
            testsd = 0
        else:
            pre_errors = None

        stop_for_errors = 0
        #? should I do some reshaping of the pre_errors?
        return weights_grad, pre_errors
    
    def backward(self, errors: cp.array) -> Tuple:
        #TODO: split the errors into pre and jump
        if self.__filter_from_next is not None:
            errors = trimed_errors(errors, self.__filter_from_next, self.neurons_shape[2])
        split_index = self.__number_of_neurons_pre
        jump_spike_per_neuron, jump_n_spike_per_neuron = self.__jump_layer.spike_trains
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__previous_layer.spike_trains
        #if padded this needs to be changed
        split_index_jump = self.__number_of_neurons_jump

        errors_pre, errors_jump = split_on_channel_dim(errors, self.neurons_shape)
        # if split_index != errors_pre.shape[1]:
        #     raise ValueError("The split index is not correct")
        # if split_index_jump != errors_jump.shape[1]:
        #     raise ValueError("The split index is not correct")
        # if the error size is to big it gives nans 
        weights_grad_pre, pre_errors_pre = self.backward_pre(errors_pre)
        #! when i put errors I get a similar type nans
        weights_grad_jump, pre_errors_jump = self.backward_jump(errors_pre)

        #problem with the input?
        #! NaNs show up here
        testing_break = "s"
        # return (weights_grad_pre, weights_grad_pre), (pre_errors_pre,pre_errors_pre)
        return (weights_grad_pre, weights_grad_jump), (pre_errors_pre, pre_errors_jump)
        

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        self.__weights_pre += delta_weights[0]
        self.__weights_jump += delta_weights[1]