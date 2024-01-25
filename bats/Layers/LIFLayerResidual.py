from operator import and_, or_
from typing import Callable, Tuple
from typing import Optional
from unittest import result
import cupy as cp
import numpy as np

from bats.AbstractLayer import AbstractLayer
from bats.CudaKernels.Wrappers.Inference import *
from bats.CudaKernels.Wrappers.Backpropagation import *


class LIFLayerResidual(AbstractLayer):
    def __init__(self, previous_layer: AbstractLayer, jump_layer: AbstractLayer, tau_s: float, theta: float, delta_theta: float,
                 weight_initializer: Callable[[int, int], cp.ndarray] = None, max_n_spike: int = 32, **kwargs):
        super().__init__(**kwargs)
        self._is_residual = True
        self.__previous_layer: AbstractLayer = previous_layer
        self.__previous_layer_residual: AbstractLayer = jump_layer
        self.__tau_s: cp.float32 = cp.float32(tau_s)
        self.__tau: cp.float32 = cp.float32(2 * tau_s)
        self.__theta_tau: cp.float32 = cp.float32(theta / self.__tau)
        self.__delta_theta_tau: cp.float32 = cp.float32(delta_theta / self.__tau)
        #TODO: Change the initial state of the weights
        if weight_initializer is None:
            self.__weights: cp.ndarray = cp.zeros((self.n_neurons, previous_layer.n_neurons + jump_layer.n_neurons), dtype=cp.float32)
            # self.__weights: cp.ndarray = cp.zeros((self.n_neurons, previous_layer.n_neurons), dtype=cp.float32)
        else:
            self.__weights: cp.ndarray = weight_initializer(self.n_neurons, previous_layer.n_neurons + jump_layer.n_neurons)
            # self.__weights: cp.ndarray = weight_initializer(self.n_neurons, previous_layer.n_neurons)
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        self.__n_spike_per_neuron: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron: Optional[cp.ndarray] = None
        self.__a: Optional[cp.ndarray] = None
        self.__x: Optional[cp.ndarray] = None
        self.__post_exp_tau: Optional[cp.ndarray] = None

        self.__pre_exp_tau_s: Optional[cp.ndarray] = None
        self.__pre_exp_tau: Optional[cp.ndarray] = None
        self.__pre_spike_weights: Optional[cp.ndarray] = None
        self.__c: Optional[cp.float32] = self.__theta_tau

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
        #? What are these two variables?
        #? How is it per neuron? (50, 784, 1)=>shape of spike_times_per_neuron on first stop of debugger
        #? It seems to be the number of spikes per layer, not per neuron, as there where 784 neurons in the input layer
        pre_spike_per_neuron_residual, pre_n_spike_per_neuron_residual = self.__previous_layer.spike_trains

        #We will try just adding them together
        jump_connection_spikes, jump_connection_spike_count = self.__previous_layer_residual.spike_trains

        #> pre_spike_per_neuron is a vector with the spike times of the previous layer

        pre_spike_per_neuron, pre_n_spike_per_neuron = fuse_inputs_append(pre_spike_per_neuron_residual, jump_connection_spikes, pre_n_spike_per_neuron_residual, jump_connection_spike_count, self.__max_n_spike)
        # join them together in a good way
        # pre_n_spike_per_neuron = np.append(pre_n_spike_per_neuron_residual, jump_connection_spike_count, axis=1)

        self.__pre_exp_tau_s, self.__pre_exp_tau = compute_pre_exps(pre_spike_per_neuron, self.__tau_s, self.__tau)
        # self.__pre_exp_tau_s_residual, self.__pre_exp_tau_residual = compute_pre_exps(pre_spike_per_neuron_residual, self.__tau_s, self.__tau)
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
            self.__n_spike_per_neuron, self.__a, self.__x, self.__spike_times_per_neuron, \
            self.__post_exp_tau = compute_spike_times(sorted_spike_times, sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                      pre_spike_weights, self.__c,
                                                      self.__delta_theta_tau,
                                                      self.__tau, cp.float32(max_simulation), self.__max_n_spike, residual = True)
            # if cp.array_equal(self.spike_trains[1], cp.zeros(self.spike_trains[1].shape)):
            #     breakpoint()
            

    def backward(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        # pre_spike_per_neuron, _ = self.__previous_layer.spike_trains

        pre_spike_per_neuron_residual, pre_n_spike_per_neuron_residual = self.__previous_layer.spike_trains

        #We will try just adding them together
        jump_connection_spikes, jump_connection_spike_count = self.__previous_layer_residual.spike_trains

        #> pre_spike_per_neuron is a vector with the spike times of the previous layer

        pre_spike_per_neuron,_ = fuse_inputs_append(pre_spike_per_neuron_residual, jump_connection_spikes, pre_n_spike_per_neuron_residual, jump_connection_spike_count, self.__max_n_spike)


        propagate_recurrent_errors(self.__x, self.__post_exp_tau, errors, self.__delta_theta_tau, residual = True)
        f1, f2 = compute_factors(self.__spike_times_per_neuron, self.__a, self.__c, self.__x,
                                 self.__post_exp_tau, self.__tau,residual=True)

        weights_grad = compute_weights_gradient(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                self.__pre_exp_tau_s, self.__pre_exp_tau, errors)
        #TODO: delay_grad = compute_delay_gradient()
        # Propagate errors
        #Here maybe I shouldn't propagate the errors to the jump connection, since the jump connection might not be trainable
        # For now I will hard code it to not propagate the errors to the jump connection

        #! VERY LIKELY THE PROBLEM IS HERE (14/01/2024)
        # if self.__previous_layer.trainable:
        # is it when the calculation is done?
        # or when the result is used?
        # Seems to be when the result is used
        if self.__previous_layer.trainable:
            pre_errors = propagate_errors_to_pre_spikes(f1, f2, self.__spike_times_per_neuron, pre_spike_per_neuron,
                                                         #!here we only feed the residual spikes
                                                        self.__pre_exp_tau_s, self.__pre_exp_tau, self.__weights,
                                                        errors, self.__tau_s, self.__tau)
        else:
            pre_errors = None

        #! I believe the problem is with weights_grad as it is used in Network.py backwards and pre_errors is not
        return weights_grad, pre_errors

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        #!Why are the last two or three entries of delta_weights nan?
        self.__weights += delta_weights

def old_fuse_inputs(residual_input, jump_input, max_n_spike, delay = None) -> cp.ndarray:
    batch_size_res, n_of_neurons_res, max_n_spike_res = residual_input.shape
    batch_size_jump, n_of_neurons_jump, max_n_spike_jump = jump_input.shape

    if batch_size_res != batch_size_jump:
        raise ValueError("Batch size of residual and jump connection must be the same.")
    if max_n_spike < max_n_spike_res or max_n_spike < max_n_spike_jump:
        raise ValueError("Max number of spikes must be greater than the max number of spikes in residual and jump connection.")


    if max_n_spike_res != max_n_spike_jump: #need to change this if
        # We pad the smallest one with inf to make them the same size
        #! Possible problem here, if inf are not ignored then I am adding a lot more spikes...
        max_n_spike = max(max_n_spike_res, max_n_spike_jump)
        residual_input = np.pad(residual_input, ((0, 0), (0, 0), (0, max_n_spike - max_n_spike_res)),constant_values = np.inf,mode = 'constant')
        jump_input = np.pad(jump_input, ((0, 0), (0, 0), (0, max_n_spike - max_n_spike_jump)), constant_values = np.inf,mode = 'constant')
    
    result = np.append(residual_input, jump_input, axis=1)
    if result.shape != (batch_size_res, n_of_neurons_res + n_of_neurons_jump, max_n_spike):
        raise ValueError("The shape of the fused is not correct.")
    return result

def fuse_inputs_append(residual_input, jump_input, count_residual, count_jump, max_n_spike, delay = None) -> Tuple[cp.ndarray, cp.ndarray]:
    batch_size_res, n_of_neurons_res, max_n_spike_res = residual_input.shape
    batch_size_jump, n_of_neurons_jump, max_n_spike_jump = jump_input.shape

    # result_count =cp.append(count_residual, count_jump, axis=1)
    result_count = cp.zeros((residual_input.shape))
    result_spikes = np.append(residual_input, jump_input, axis=1)
    if cp.any(result_count > max_n_spike):
        raise ValueError("The count of spikes is greater than the max number of spikes")
    # result_count = count_residual
    # result_spikes = residual_input
    return result_spikes, result_count
    # We n


def fuse_inputs(residual_input, jump_input, count_residual, count_jump, max_n_spike, delay = None) -> Tuple[cp.ndarray, cp.ndarray]:
    #! Illegal memory error still occurs even without this code

    #! how does it handle the spikes when there are too many?
    #! buffer should check all the spikes and if too many it crashes
    # HOW DOES THE spike buffer break?
    # make it break and see what happens
    
    # if delay is None:
    #     #by default the delay is the mean of the residual input,
    #     delay = cp.mean(residual_input[np.isfinite(residual_input)])
    #     # delay = 0
    # out = cp.empty(jump_input.shape, dtype=int)
    # out[out == 0] = delay
    # cp.add(jump_input, out, out = jump_input)
    result_count = cp.maximum(count_residual, count_jump)

    batch_size_res, n_of_neurons_res, max_n_spike_res = residual_input.shape
    batch_size_jump, n_of_neurons_jump, max_n_spike_jump = jump_input.shape

    not_inf_mask_res = cp.logical_not(cp.isinf(residual_input))
    not_inf_mask_jump = cp.logical_not(cp.isinf(jump_input))

    inf_mask_res = cp.isinf(residual_input)
    inf_mask_jump = cp.isinf(jump_input)

    xor_combined_mask = cp.logical_xor(not_inf_mask_res, not_inf_mask_jump)
    or_combined_mask = cp.logical_or(not_inf_mask_res, not_inf_mask_jump)
    and_combined_mask = cp.logical_and(not_inf_mask_res, not_inf_mask_jump)

    #! for now if both are inf we take residual, we should take whichever is not inf
    get_non_infinite = cp.where(inf_mask_res, jump_input, residual_input)
    get_non_infinite = cp.where(inf_mask_jump, residual_input, get_non_infinite)
    result_times = cp.where(or_combined_mask, jump_input, residual_input)

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
                      cp.mean(cp.array([ residual_input, residual_input ]), axis=0)
                      )


    return result_times, result_count
