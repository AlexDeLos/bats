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
                 weight_initializer: Callable[[int, int], cp.ndarray] = None, fuse_function = "Append", max_n_spike: int = 32, **kwargs): # type: ignore
        super().__init__(**kwargs)
        self._is_residual = True
        self.__previous_layer: AbstractLayer = previous_layer
        self.__jump_layer: AbstractLayer = jump_layer
        self.__tau_s_res: cp.float32 = cp.float32(tau_s)
        self.__tau_s_jump: cp.float32 = cp.float32(tau_s)
        self.__tau_res: cp.float32 = cp.float32(2 * tau_s)
        self.__tau_jump: cp.float32 = cp.float32(2 * tau_s)
        self.__theta_tau_res: cp.float32 = cp.float32(theta / self.__tau_res) # type: ignore
        self.__theta_tau_jump: cp.float32 = cp.float32(theta / self.__tau_jump) # type: ignore
        self.__delta_theta_tau_res: cp.float32 = cp.float32(delta_theta / self.__tau_res) # type: ignore
        self.__delta_theta_tau_jump: cp.float32 = cp.float32(delta_theta / self.__tau_jump) # type: ignore
        self.__fuse_function = fuse_function
        #TODO: Change the initial state of the weights
        if weight_initializer is None:
            if fuse_function == "Append": 
                self.__weights_res: cp.ndarray = cp.zeros((int(cp.ceil(self.n_neurons/2)), previous_layer.n_neurons), dtype=cp.float32) # type: ignore
                self.__weights_jump: cp.ndarray = cp.zeros((int(cp.floor(self.n_neurons/2)), jump_layer.n_neurons), dtype=cp.float32) # type: ignore
            else:
                self.__weights_res: cp.ndarray = cp.zeros((int(cp.ceil(self.n_neurons/2)), previous_layer.n_neurons), dtype=cp.float32) # type: ignore
                self.__weights_jump: cp.ndarray = cp.zeros((int(cp.floor(self.n_neurons/2)), jump_layer.n_neurons), dtype=cp.float32) # type: ignore

        else:
            if fuse_function == "Append":
                self.__weights_res: cp.ndarray = weight_initializer(int(cp.floor(self.n_neurons/2)), previous_layer.n_neurons)
                self.__weights_jump: cp.ndarray = weight_initializer(int(cp.ceil(self.n_neurons/2)), jump_layer.n_neurons)

            else:
                self.__weights_res: cp.ndarray = weight_initializer(int(cp.floor(self.n_neurons/2)), previous_layer.n_neurons)
                self.__weights_jump: cp.ndarray = weight_initializer(int(cp.ceil(self.n_neurons/2)), jump_layer.n_neurons)
        self.__max_n_spike: cp.int32 = cp.int32(max_n_spike)

        self.__n_spike_per_neuron_res: Optional[cp.ndarray] = None
        self.__n_spike_per_neuron_jump: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron_res: Optional[cp.ndarray] = None
        self.__spike_times_per_neuron_jump: Optional[cp.ndarray] = None
        self.__a_res: Optional[cp.ndarray] = None
        self.__a_jump: Optional[cp.ndarray] = None
        self.__x_res: Optional[cp.ndarray] = None
        self.__x_jump: Optional[cp.ndarray] = None
        self.__post_exp_tau_res: Optional[cp.ndarray] = None
        self.__post_exp_tau_jump: Optional[cp.ndarray] = None

        self.__pre_exp_tau_s_res: Optional[cp.ndarray] = None
        self.__pre_exp_tau_s_jump: Optional[cp.ndarray] = None
        self.__pre_exp_tau_res: Optional[cp.ndarray] = None
        self.__pre_exp_tau_jump: Optional[cp.ndarray] = None
        # self.__pre_spike_weights: Optional[cp.ndarray] = None
        self.__c_res: Optional[cp.float32] = self.__theta_tau_res
        self.__c_jump: Optional[cp.float32] = self.__theta_tau_jump

    @property
    def trainable(self) -> bool:
        return True

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        # testing = self.__spike_times_per_neuron_res
        # testing2 = self.__spike_times_per_neuron_jump
        #first half are residual, second half are jump
        if self.__fuse_function == "Append":
            res =  fuse_inputs_append(self.__spike_times_per_neuron_res, self.__spike_times_per_neuron_jump, self.__n_spike_per_neuron_res, self.__n_spike_per_neuron_jump, self.__max_n_spike)
        else:
            #! shape is different here than in the other option don;t belive it fits with n_neurons
            res = fuse_inputs(self.__spike_times_per_neuron_res, self.__spike_times_per_neuron_jump, self.__n_spike_per_neuron_res, self.__n_spike_per_neuron_jump, self.__max_n_spike)
        return res
        return self.__spike_times_per_neuron_res, self.__n_spike_per_neuron_res

    @property
    def weights(self) -> Tuple[Optional[cp.ndarray], Optional[cp.ndarray]]:
        return (self.__weights_res, self.__weights_jump)

    @weights.setter
    def weights(self, weights: Tuple[cp.ndarray, cp.ndarray]) -> None:
        self.__weights_res = cp.array(weights, dtype=cp.float32)
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

    # forward function for the jump part of the residual layer
    def forward_res(self, max_simulation: float, training: bool = False) -> None:
        #? What are these two variables?
        #? How is it per neuron? (50, 784, 1)=>shape of spike_times_per_neuron on first stop of debugger
        #? It seems to be the number of spikes per layer, not per neuron, as there where 784 neurons in the input layer
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__previous_layer.spike_trains

        self.__pre_exp_tau_s_res, self.__pre_exp_tau_res = compute_pre_exps(pre_spike_per_neuron, self.__tau_s_res, self.__tau_res)
        # END OF PREVIOUS LAYER INPUTS

        # Sort spikes for inference
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(pre_spike_per_neuron,
                                                                                    pre_n_spike_per_neuron)
        if sorted_indices.size == 0:  # No input spike in the batch
            batch_size = pre_spike_per_neuron.shape[0]
            shape = (batch_size, int(cp.floor(self.n_neurons/2)), self.__max_n_spike)
            self.__n_spike_per_neuron_res = cp.zeros((batch_size, int(cp.floor(self.n_neurons/2))), dtype=cp.int32)
            self.__spike_times_per_neuron_res = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__post_exp_tau_res = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__a_res = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__x_res = cp.full(shape, cp.inf, dtype=cp.float32)
        else:
            #? What are they sorting the spikes by? TIME obviously
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)

            # Store the times of the spikes (this array A[i]<A[i+1] for all i)
            sorted_spike_times[sorted_indices == -1] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_s_res, new_shape), sorted_indices, axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_res, new_shape), sorted_indices, axis=1)
            pre_spike_weights = get_spike_weights(self.weights[0], sorted_spike_indices)

            #! self.weights has nans in it
            # it's shape is (100, 240)
            #! pre spike weights has nans in it
            # Compute spikes, everything has been calculated in order to make this
            self.__n_spike_per_neuron_res, self.__a_res, self.__x_res, self.__spike_times_per_neuron_res, \
            self.__post_exp_tau_res = compute_spike_times(sorted_spike_times, sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                      pre_spike_weights, self.__c_res,
                                                      self.__delta_theta_tau_res,
                                                      self.__tau_res, cp.float32(max_simulation), self.__max_n_spike)
            testing = self.__spike_times_per_neuron_res
            cp.where(cp.isnan(testing))
            cp.any(cp.isnan(testing))
            sdfds = 'f'
    # forward function for the jump part of the residual layer
    def forward_jump(self, max_simulation: float, training: bool = False) -> None:
        #? What are these two variables?
        #? How is it per neuron? (50, 784, 1)=>shape of spike_times_per_neuron on first stop of debugger
        #? It seems to be the number of spikes per layer, not per neuron, as there where 784 neurons in the input layer
        pre_spike_per_neuron, pre_n_spike_per_neuron = self.__jump_layer.spike_trains
        #? is this where pre_esp_tau_s_jump is given a bad shape?
        self.__pre_exp_tau_s_jump, self.__pre_exp_tau_jump = compute_pre_exps(pre_spike_per_neuron, self.__tau_s_jump, self.__tau_jump)
        # END OF PREVIOUS LAYER INPUTS

        # Sort spikes for inference
        new_shape, sorted_indices, spike_times_reshaped = get_sorted_spikes_indices(pre_spike_per_neuron,
                                                                                    pre_n_spike_per_neuron)
        if sorted_indices.size == 0:  # No input spike in the batch
            batch_size = pre_spike_per_neuron.shape[0]
            shape = (batch_size, int(cp.ceil(self.n_neurons/2)), self.__max_n_spike)
            self.__n_spike_per_neuron_jump = cp.zeros((batch_size, int(cp.ceil(self.n_neurons/2))), dtype=cp.int32)
            self.__spike_times_per_neuron_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__post_exp_tau_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__a_jump = cp.full(shape, cp.inf, dtype=cp.float32)
            self.__x_jump = cp.full(shape, cp.inf, dtype=cp.float32)
        else:
            #? What are they sorting the spikes by? TIME obviously
            sorted_spike_indices = (sorted_indices.astype(cp.int32) // pre_spike_per_neuron.shape[2])
            sorted_spike_times = cp.take_along_axis(spike_times_reshaped, sorted_indices, axis=1)

            # Store the times of the spikes (this array A[i]<A[i+1] for all i)
            sorted_spike_times[sorted_indices == -1] = cp.inf
            sorted_pre_exp_tau_s = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_s_jump, new_shape), sorted_indices, axis=1)
            sorted_pre_exp_tau = cp.take_along_axis(cp.reshape(self.__pre_exp_tau_jump, new_shape), sorted_indices, axis=1)
            pre_spike_weights = get_spike_weights(self.weights[1], sorted_spike_indices)

            # Compute spikes, everything has been calculated in order to make this
            self.__n_spike_per_neuron_jump, self.__a_jump, self.__x_jump, self.__spike_times_per_neuron_jump, \
            self.__post_exp_tau_jump = compute_spike_times(sorted_spike_times, sorted_pre_exp_tau_s, sorted_pre_exp_tau,
                                                      pre_spike_weights, self.__c_jump,
                                                      self.__delta_theta_tau_jump,
                                                      self.__tau_jump, cp.float32(max_simulation), self.__max_n_spike)
            testing = self.__spike_times_per_neuron_res
            cp.any(cp.isnan(testing))
            sdfds = 'f'
    
    def forward(self, max_simulation: float, training: bool = False) -> None:
        self.forward_res(max_simulation, training)
        self.forward_jump(max_simulation, training)

    # backwards function for the jump part of the residual layer
    def backward_jump(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        pre_spike_per_neuron, _ = self.__jump_layer.spike_trains
        # pre_spike_per_neuron, _ = self.__previous_layer.spike_trains
        #! this call changes the errors and adds nans to it
        x_jump = self.__x_jump
        post_exp_tau_jump = self.__post_exp_tau_jump
        delta_theta_tau_jump = self.__delta_theta_tau_jump
        propagate_recurrent_errors(self.__x_jump, self.__post_exp_tau_jump, errors, self.__delta_theta_tau_jump)
        f1, f2 = compute_factors(self.__spike_times_per_neuron_jump, self.__a_jump, self.__c_jump, self.__x_jump,
                                 self.__post_exp_tau_jump, self.__tau_jump)

        spike_times_per_neuron = self.__spike_times_per_neuron_jump
        pre_exp_tau_s = self.__pre_exp_tau_s_jump # -> this has the wrong shape
        pre_exp_tau = self.__pre_exp_tau_jump
        # none of the inputs have nans
        # error must be in the shapes
        weights_grad = compute_weights_gradient(f1, f2, self.__spike_times_per_neuron_jump, pre_spike_per_neuron,
                                                self.__pre_exp_tau_s_jump, self.__pre_exp_tau_jump, errors)
        nan_indices = cp.where(cp.isnan(weights_grad))
        #: The nans have a different shape each time, so might nit be a shape problem

        # Propagate errors
        # what are the dimensions of f1
        # what are the dimensions of pre_exp_tau_s
        if self.__jump_layer.trainable:
            pre_errors = propagate_errors_to_pre_spikes(f1, f2, self.__spike_times_per_neuron_jump, pre_spike_per_neuron,
                                                        self.__pre_exp_tau_s_jump, self.__pre_exp_tau_jump, self.__weights_jump,
                                                        errors, self.__tau_s_jump, self.__tau_jump)
        else:
            pre_errors = None
        # pre_spike_per_neuron.shape = (20, 200, 30)
        return weights_grad, pre_errors
    
    def backward_res(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        # Compute gradient
        pre_spike_per_neuron, _ = self.__previous_layer.spike_trains
        x_res = self.__x_res
        post_exp_tau_res = self.__post_exp_tau_res
        delta_theta_tau_res = self.__delta_theta_tau_res
        propagate_recurrent_errors(self.__x_res, self.__post_exp_tau_res, errors, self.__delta_theta_tau_res)
        f1, f2 = compute_factors(self.__spike_times_per_neuron_res, self.__a_res, self.__c_res, self.__x_res,
                                 self.__post_exp_tau_res, self.__tau_res)

        weights_grad = compute_weights_gradient(f1, f2, self.__spike_times_per_neuron_res, pre_spike_per_neuron,
                                                self.__pre_exp_tau_s_res, self.__pre_exp_tau_res, errors)
        
        # Propagate errors
        # what are the dimensions of f1
        # what are the dimensions of pre_exp_tau_s
        if self.__previous_layer.trainable:
            pre_errors = propagate_errors_to_pre_spikes(f1, f2, self.__spike_times_per_neuron_res, pre_spike_per_neuron,
                                                        self.__pre_exp_tau_s_res, self.__pre_exp_tau_res, self.__weights_res,
                                                        errors, self.__tau_s_res, self.__tau_res)
        else:
            pre_errors = None
        # pre_spike_per_neuron.shape = (20, 240, 30)
        return weights_grad, pre_errors

    def backward(self, errors: cp.array) -> Optional[Tuple[cp.ndarray, cp.ndarray]]: # type: ignore
        # Compute gradient
        # errors_save = errors.copy()

        #! weights_grad_jump has nans in it
        #! check the shapes pf everything
        #! i need to split the errors
        # the order of this split has been checked and is correct the first half is residual and the second half is jump
        errors_res, errors_jump = cp.split(errors, 2, axis=1)
        #DONE: flip the inputs, where flipped to test ->the problem is not the input
        #! problem is with the function, not the inputs:
        # The problem was that in the forward I took the res input not the jump
        weights_grad_jump, pre_errors_jump = self.backward_jump(errors_jump) # type: ignore
        weights_grad_res, pre_errors_res = self.backward_res(errors_res) # type: ignore
        # weights_grad = cp.divide(cp.add(weights_grad_jump, weights_grad_res),2)
        return (weights_grad_res, weights_grad_jump), (pre_errors_res,pre_errors_jump)

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        #!Why are the last two or three entries of delta_weights nan?
        #? I could allow this to be seperate
        # TODO: to make this separate I need to make the gradient separate instead of averaging them in the backward function
        # delta_split = cp.split(delta_weights, 2, axis=1)
        self.__weights_res += delta_weights[0]
        #! delta_weights[1] has nans in it
        self.__weights_jump += delta_weights[1]
    
    def store(self, dir_path: Path) -> None:
        weights = self.weights
        if weights is not None:
            pre,_ = WEIGHTS_FILE_SUFFIX.split('.npy')
            filename_res = dir_path / (self._name + pre + "_res" + '.npy')
            filename_jump = dir_path / (self._name + pre + "_jump" + '.npy')
            np.save(filename_res, self.__weights_res.get())
            np.save(filename_jump, self.__weights_jump.get())

    def restore(self, dir_path: Path) -> None:
        pre,_ = WEIGHTS_FILE_SUFFIX.split('.npy')
        filename_res = dir_path / (self._name + pre + "_res" + '.npy')
        filename_jump = dir_path / (self._name + pre + "_jump" + '.npy')
        if filename_res.exists():
            self.__weights_res = np.load(filename_res)
        if filename_jump.exists():
            self.__weights_jump = np.load(filename_jump)



def fuse_inputs_append(residual_input, jump_input, count_residual, count_jump, max_n_spike, delay = None) -> Tuple[cp.ndarray, cp.ndarray]:
    batch_size_res, n_of_neurons_res, max_n_spike_res = residual_input.shape
    batch_size_jump, n_of_neurons_jump, max_n_spike_jump = jump_input.shape

    result_count =cp.append(count_residual, count_jump, axis=1)
    # this changes the effect
    # result_count = count_residual
    # result_count = cp.zeros((residual_input.shape))
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



# def old_fuse_inputs(residual_input, jump_input, max_n_spike, delay = None) -> cp.ndarray:
#     batch_size_res, n_of_neurons_res, max_n_spike_res = residual_input.shape
#     batch_size_jump, n_of_neurons_jump, max_n_spike_jump = jump_input.shape

#     if batch_size_res != batch_size_jump:
#         raise ValueError("Batch size of residual and jump connection must be the same.")
#     if max_n_spike < max_n_spike_res or max_n_spike < max_n_spike_jump:
#         raise ValueError("Max number of spikes must be greater than the max number of spikes in residual and jump connection.")


#     if max_n_spike_res != max_n_spike_jump: #need to change this if
#         # We pad the smallest one with inf to make them the same size
#         #! Possible problem here, if inf are not ignored then I am adding a lot more spikes...
#         max_n_spike = max(max_n_spike_res, max_n_spike_jump)
#         residual_input = np.pad(residual_input, ((0, 0), (0, 0), (0, max_n_spike - max_n_spike_res)),constant_values = np.inf,mode = 'constant')
#         jump_input = np.pad(jump_input, ((0, 0), (0, 0), (0, max_n_spike - max_n_spike_jump)), constant_values = np.inf,mode = 'constant')
    
#     result = np.append(residual_input, jump_input, axis=1)
#     if result.shape != (batch_size_res, n_of_neurons_res + n_of_neurons_jump, max_n_spike):
#         raise ValueError("The shape of the fused is not correct.")
#     return result