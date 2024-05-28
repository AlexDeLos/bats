from typing import Callable, Tuple
from typing import Optional
import cupy as cp
import numpy as np
from pathlib import Path

from bats.AbstractLayer import AbstractLayer
from bats.CudaKernels.Wrappers.Inference import *
from bats.CudaKernels.Wrappers.Backpropagation import *

WEIGHTS_FILE_SUFFIX = "_weights.npy"

class LIFLayerResidual(AbstractLayer):
    def __init__(self, previous_layer: AbstractLayer, jump_layer: AbstractLayer, tau_s: float, theta: float, delta_theta: float,
                 weight_initializer: Callable[[int, int], cp.ndarray] = None, fuse_function = "Append", use_delay = False, max_n_spike: int = 32, **kwargs): # type: ignore
        super().__init__(**kwargs, is_residual=True)
        self._is_residual = True
        self.use_delay = use_delay
        self.__previous_layer: AbstractLayer = previous_layer
        self.__jump_layer: AbstractLayer = jump_layer
        self.__tau_s_res: cp.float32 = cp.float32(tau_s)
        self.__tau_res: cp.float32 = cp.float32(2 * tau_s)
        self.__theta_tau_res: cp.float32 = cp.float32(theta / self.__tau_res) # type: ignore
        self.__delta_theta_tau_res: cp.float32 = cp.float32(delta_theta / self.__tau_res) # type: ignore
        self.__fuse_function = fuse_function
        #! for testing REMOVE:
        # Change teh size of the weights
        #TODO: Change the initial state of the weights
        if weight_initializer is None:
            if fuse_function == "Append": 
                self.__weights_res: cp.ndarray = cp.zeros((self.n_neurons, previous_layer.n_neurons+jump_layer.n_neurons), dtype=cp.float32) # type: ignore
            else:
                self.__weights_res: cp.ndarray = cp.zeros((self.n_neurons, previous_layer.n_neurons), dtype=cp.float32) # type: ignore

        else:
            if fuse_function == "Append":
                self.__weights_res: cp.ndarray = weight_initializer(self.n_neurons, previous_layer.n_neurons + jump_layer.n_neurons) # type: ignore


            else:
                if (jump_layer.n_neurons) != previous_layer.n_neurons:
                    raise ValueError("The number of neurons in the previous layer and the jump layer must be the same for the average fusion function")
                self.__weights_res: cp.ndarray = weight_initializer(self.n_neurons, previous_layer.n_neurons) # type: ignore
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        # self.__pre_spike_weights: Optional[cp.ndarray] = None
        self.__c_res: Optional[cp.float32] = self.__theta_tau_res

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        #first half are residual, second half are jump
        spikes = self.__spike_times_per_neuron
        count = self.__n_spike_per_neuron
        res = (spikes, count)
        return res

    @property
    def weights(self) -> Tuple[Optional[cp.ndarray], Optional[cp.ndarray]]:
        return self.__weights_res
        
    @weights.setter
    def weights(self, weights: Tuple[cp.ndarray, cp.ndarray]) -> None:
        self.__weights_res = cp.array(weights, dtype=cp.float32)

    @property
    def previous_layer(self) -> AbstractLayer:
        return self.__previous_layer
    
    @property
    def jump_layer(self) -> AbstractLayer:
        return self.__jump_layer
    
    @property
    def fuse_function(self) -> str:
        return self.__fuse_function

    def reset(self) -> None:
        self.__n_spike_per_neuron_res = None
        self.__spike_times_per_neuron_res = None
        self.__pre_exp_tau_s_res = None
        self.__pre_exp_tau_res = None
        # self.__pre_spike_weights = None
        self.__a_res = None
        self.__x_res = None
        self.__post_exp_tau_res = None


    def forward(self, max_simulation: float, training: bool = False) -> None:
        pre_spike_per_neuron1, pre_n_spike_per_neuron1 = self.__previous_layer.spike_trains
        pre_spike_per_neuron2, pre_n_spike_per_neuron2 = self.__jump_layer.spike_trains
        if cp.any(cp.isnan(pre_spike_per_neuron1)):
            raise ValueError("Nans in the previous layer input")
        if cp.any(cp.isnan(pre_spike_per_neuron2)):
            raise ValueError("Nans in the jump layer input")
        # pre_spike_per_neuron2 = cp.full(pre_spike_per_neuron2.shape, cp.inf)
        # pre_n_spike_per_neuron2 = cp.zeros(pre_n_spike_per_neuron2.shape, dtype=cp.int32)
        new_max_n_spike = max(pre_spike_per_neuron1.shape[2], pre_spike_per_neuron2.shape[2])
        if self.__fuse_function == "Append":
            pre_spike_per_neuron, pre_n_spike_per_neuron = fuse_inputs_append(pre_spike_per_neuron1, pre_spike_per_neuron2, pre_n_spike_per_neuron1, pre_n_spike_per_neuron2, new_max_n_spike, self.use_delay)
        elif self.__fuse_function == "Stack":
            pre_spike_per_neuron, pre_n_spike_per_neuron = fuse_inputs_stack(pre_spike_per_neuron1, pre_spike_per_neuron2, pre_n_spike_per_neuron1, pre_n_spike_per_neuron2, new_max_n_spike, self.use_delay)
        else:
            pre_spike_per_neuron, pre_n_spike_per_neuron = fuse_inputs(pre_spike_per_neuron1, pre_spike_per_neuron2, pre_n_spike_per_neuron1, pre_n_spike_per_neuron2, new_max_n_spike, self.use_delay)

        if cp.any(cp.isnan(pre_spike_per_neuron)):
            raise ValueError("Nans in the fused input")
        if cp.any(cp.isnan(pre_n_spike_per_neuron)):
            raise ValueError("Nans in the fused input count")
        self.__pre_spike_trains = (pre_spike_per_neuron, pre_n_spike_per_neuron)

        self.__pre_exp_tau_s, self.__pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s_res, self.__tau_res)
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
                                                    pre_spike_weights, self.__c_res,
                                                    self.__delta_theta_tau_res,
                                                    self.__tau_res, cp.float32(max_simulation), self.__max_n_spike)
            spikes = self.__spike_times_per_neuron
            count = self.__n_spike_per_neuron
            t = 'f'

    def backward(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]: # type: ignore
        # The problem was that in the forward I took the res input not the jump
        if self.__fuse_function == "Append":
            weights_grad, pre_errors = self.backward_append(errors) # type: ignore
            #! split the errors
            pre_errors_res, pre_errors_jump=cp.split(pre_errors, 2, axis=1)
            # weights_grad_res, weights_grad_jump = cp.split(weights_grad, 2, axis=1)
            return weights_grad, (pre_errors_res, pre_errors_jump)
        #! problem with the errors
        else:
            weights_grad, pre_errors = self.backward_append(errors) 
            return weights_grad, pre_errors

    def backward_append(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        pre_spike_per_neuron, _ = self.__pre_spike_trains
        propagate_recurrent_errors(self.__x, self.__post_exp_tau, errors, self.__delta_theta_tau_res)# all the shape of this layer
        f1, f2 = compute_factors(self.__spike_times_per_neuron, self.__a, self.__c_res, self.__x,
                                 self.__post_exp_tau, self.__tau_res)
        #! nans show up here when getting the previous layer spikes
        test = self.__spike_times_per_neuron
        weights_grad = compute_weights_gradient(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                self.__pre_exp_tau_s, self.__pre_exp_tau, errors)
        # Propagate errors
        # what are the dimensions of f1
        # what are the dimensions of pre_exp_tau_s
        if self.__previous_layer.trainable:
            pre_errors = propagate_errors_to_pre_spikes(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                        self.__pre_exp_tau_s, self.__pre_exp_tau, self.__weights_res,
                                                        errors, self.__tau_s_res, self.__tau_res)
            asdasd = 0
        else:
            pre_errors = None

        asddas= ''
        return weights_grad, pre_errors
    
    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        #!Why are the last two or three entries of delta_weights nan?
        #? I could allow this to be seperate
        # TODO: to make this separate I need to make the gradient separate instead of averaging them in the backward function

        self.__weights_res += delta_weights
    
    def store(self, dir_path: Path) -> None:
        weights = self.weights
        if weights is not None:
            pre,_ = WEIGHTS_FILE_SUFFIX.split('.npy')
            filename_res = dir_path / (self._name + pre + "_res" + '.npy')
            np.save(filename_res, self.__weights_res.get())

    def restore(self, dir_path: Path) -> None:
        pre,_ = WEIGHTS_FILE_SUFFIX.split('.npy')
        filename_res = dir_path / (self._name + pre + "_res" + '.npy')
        filename_jump = dir_path / (self._name + pre + "_jump" + '.npy')
        if filename_res.exists():
            self.__weights_res = np.load(filename_res)



def fuse_inputs_append(pre_input, jump_input, count_pre, count_jump, max_n_spike, delay = False) -> Tuple[cp.ndarray, cp.ndarray]:
    # batch_size_res, n_of_neurons_res, max_n_spike_res = residual_input.shape
    # batch_size_jump, n_of_neurons_jump, max_n_spike_jump = jump_input.shape
    if cp.any(cp.isnan(pre_input)):
        raise ValueError("Nans in the pre input")
    if cp.any(cp.isnan(jump_input)):
        raise ValueError("Nans in the jump input")
    og_jump_input = cp.copy(jump_input)
    og_pre_input = cp.copy(pre_input)
    # check if all elements of one of the 2 arrays are inf
    pre_is_inf = cp.all(cp.isinf(pre_input))
    jump_is_inf = cp.all(cp.isinf(jump_input))
    if delay and not(pre_is_inf or jump_is_inf):
        copy_pre_spike_per_neuron = cp.copy(pre_input)
        non_inf_values_pre = copy_pre_spike_per_neuron[cp.isfinite(copy_pre_spike_per_neuron)]  # Select non-inf values
        average_non_inf_pre = cp.mean(non_inf_values_pre)
        copy_jump_spike_per_neuron = cp.copy(jump_input)
        non_inf_values_jump = copy_jump_spike_per_neuron[cp.isfinite(copy_jump_spike_per_neuron)]
        average_non_inf_jump = cp.mean(non_inf_values_jump)
        time_delay = average_non_inf_pre - average_non_inf_jump
        jump_input = jump_input + time_delay
    result_count =cp.append(count_pre, count_jump, axis=1)
    # this changes the effect
    # result_count = count_residual
    # result_count = cp.zeros((residual_input.shape))
    if jump_input.shape[2] != pre_input.shape[2]:
        jump_input = cp.pad(jump_input, ((0, 0), (0, 0), (0, pre_input.shape[2] - jump_input.shape[2])), mode='constant', constant_values=cp.inf)
    result_spikes = np.append(pre_input, jump_input, axis=1)
    if cp.any(result_count > max_n_spike):
        raise ValueError("The count of spikes is greater than the max number of spikes")
    # result_count = count_residual
    # result_spikes = residual_input
    if cp.any(cp.isnan(result_spikes)):
        raise ValueError("Nans in the fused input")
    return result_spikes, result_count

def fuse_inputs_stack(pre_input, jump_input, count_residual, count_jump, max_n_spike, delay = False) -> Tuple[cp.ndarray, cp.ndarray]:
    # in this function we add the 2 inputs together
    if delay:
        copy_pre_spike_per_neuron = cp.copy(pre_input)
        non_inf_values_pre = copy_pre_spike_per_neuron[cp.isfinite(copy_pre_spike_per_neuron)]  # Select non-inf values
        average_non_inf_pre = cp.mean(non_inf_values_pre)
        copy_jump_spike_per_neuron = cp.copy(jump_input)
        non_inf_values_jump = copy_jump_spike_per_neuron[cp.isfinite(copy_jump_spike_per_neuron)]
        average_non_inf_jump = cp.mean(non_inf_values_jump)
        time_delay = average_non_inf_pre - average_non_inf_jump
        jump_input = jump_input+ time_delay
    count = count_residual + count_jump
    result = cp.concatenate([pre_input, jump_input], axis=2)
    # sort the spikes on the last axis of the result
    result = cp.sort(result, axis=2)
    return result, count

def fuse_inputs(pre_input, jump_input, count_residual, count_jump, max_n_spike, delay = False) -> Tuple[cp.ndarray, cp.ndarray]:
    #! Illegal memory error still occurs even without this code

    #! how does it handle the spikes when there are too many?
    #! buffer should check all the spikes and if too many it crashes
    # HOW DOES THE spike buffer break?
    # make it break and see what happens
    
    if delay:
        copy_pre_spike_per_neuron = cp.copy(pre_input)
        non_inf_values_pre = copy_pre_spike_per_neuron[cp.isfinite(copy_pre_spike_per_neuron)]  # Select non-inf values
        average_non_inf_pre = cp.mean(non_inf_values_pre)
        copy_jump_spike_per_neuron = cp.copy(jump_input)
        non_inf_values_jump = copy_jump_spike_per_neuron[cp.isfinite(copy_jump_spike_per_neuron)]
        average_non_inf_jump = cp.mean(non_inf_values_jump)
        time_delay = average_non_inf_pre - average_non_inf_jump
        jump_input = jump_input+ time_delay

    result_count = cp.maximum(count_residual, count_jump)

    batch_size_res, n_of_neurons_res, max_n_spike_res = pre_input.shape
    batch_size_jump, n_of_neurons_jump, max_n_spike_jump = jump_input.shape

    not_inf_mask_res = cp.logical_not(cp.isinf(pre_input))
    not_inf_mask_jump = cp.logical_not(cp.isinf(jump_input))

    inf_mask_res = cp.isinf(pre_input)
    inf_mask_jump = cp.isinf(jump_input)

    xor_combined_mask = cp.logical_xor(not_inf_mask_res, not_inf_mask_jump)
    or_combined_mask = cp.logical_or(not_inf_mask_res, not_inf_mask_jump)
    and_combined_mask = cp.logical_and(not_inf_mask_res, not_inf_mask_jump)

    #! for now if both are inf we take residual, we should take whichever is not inf
    get_non_infinite = cp.where(inf_mask_res, jump_input, pre_input)
    get_non_infinite = cp.where(inf_mask_jump, pre_input, get_non_infinite)
    result_times = cp.where(or_combined_mask, jump_input, pre_input)

    #if true in mask then take the value, else take average
    # result = cp.where(or_combined_mask, residual_input, jump_input)
    
    # we make the average of both inputs
    if batch_size_res != batch_size_jump:
        raise ValueError("The batch size of the residual and jump input must be the same")
    if n_of_neurons_res != n_of_neurons_jump:
        raise ValueError("The number of neurons of the residual and jump input must be the same")
    if max_n_spike_res != max_n_spike_jump:
        raise ValueError("The max number of spikes of the residual and jump input must be the same")
    # return residual_input
    result_times = cp.where(xor_combined_mask,
                        get_non_infinite,
                      cp.mean(cp.array([ pre_input, pre_input ]), axis=0))

    return result_times, result_count