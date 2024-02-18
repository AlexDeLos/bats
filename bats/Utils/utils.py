import re
import cupy as cp
from math import sqrt
# This file contains utility functions that are used in the main code

def split_on_channel_dim(errors, shape):
    #! how are the channels saved in the errors?
    x, y, c = shape.get()
    errors = cp.reshape(errors, (errors.shape[0], x, y, c*2, errors.shape[2]))
    errors_pre, errors_jump = cp.split(errors, 2, axis=3)
    errors_pre = cp.reshape(errors_pre, (errors.shape[0], x*y*c, errors.shape[-1]))
    errors_jump = cp.reshape(errors_jump, (errors.shape[0], x*y*c, errors.shape[-1]))
    return errors_pre, errors_jump
 
def aped_on_channel_dim(pre_spike_per_neuron, pre_n_spike_per_neuron, jump_spike_per_neuron, jump_n_spike_per_neuron, pre_shape, jump_shape):
    batch_size, spikes, max_n_spikes = pre_spike_per_neuron.shape
    jump_batch_size, jump_spikes, jump_max_n_spikes = jump_spike_per_neuron.shape
    if batch_size != jump_batch_size:
        raise RuntimeError("The batch sizes of the two inputs are not the same")
    pre_x, pre_y, pre_c = pre_shape.get()
    jump_x, jump_y, jump_c = jump_shape.get()
    if pre_x != jump_x or pre_y != jump_y:
        raise RuntimeError("The input shapes are not the same")
    if spikes != pre_x * pre_y * pre_c or jump_spikes != jump_x * jump_y * jump_c:
        raise RuntimeError("The number of spikes is not the same as the number of neurons")
    
    if max_n_spikes != jump_max_n_spikes:
        print('Warning: the maximum number of spikes is not the same for the two inputs')
        max_n_spikes = max(max_n_spikes, jump_max_n_spikes)
        # now we make sure that the two inputs have the same number of spikes by adding 0s to the smaller one
        # TODO: does reshape do this?-> NO
        padding_for_max_spikes_pre = ([0,0],[0,0], [0,max_n_spikes-pre_spike_per_neuron.shape[2]])
        padding_for_max_spikes_jump = ([0,0],[0,0], [0,max_n_spikes-jump_spike_per_neuron.shape[2]])
        pre_spike_per_neuron = cp.pad(pre_spike_per_neuron, padding_for_max_spikes_pre, mode='constant', constant_values=cp.inf)
        jump_spike_per_neuron = cp.pad(jump_spike_per_neuron, padding_for_max_spikes_jump, mode='constant', constant_values=cp.inf)

    #? Am I using the right reshape order?
    pre_spike_per_neuron = cp.reshape(pre_spike_per_neuron, (batch_size,pre_x, pre_y, pre_c, max_n_spikes))
    jump_spike_per_neuron = cp.reshape(jump_spike_per_neuron, (jump_batch_size,jump_x, jump_y, jump_c, jump_max_n_spikes))
    pre_n_spike_per_neuron = cp.reshape(pre_n_spike_per_neuron, (batch_size,pre_x, pre_y, pre_c))
    jump_n_spike_per_neuron = cp.reshape(jump_n_spike_per_neuron, (jump_batch_size,jump_x, jump_y, jump_c))

    # Now we append pre_spike_per_neuron and jump_spike_per_neuron on the channel dimension
    new_spike_per_neuron = cp.append(pre_spike_per_neuron, jump_spike_per_neuron, axis=3)
    new_spike_per_neuron = cp.reshape(new_spike_per_neuron, (batch_size, pre_x*pre_y*(pre_c+jump_c), max_n_spikes))

    new_n_spike_per_neuron = cp.append(pre_n_spike_per_neuron, jump_n_spike_per_neuron, axis=3)
    new_n_spike_per_neuron = cp.reshape(new_n_spike_per_neuron, (batch_size, pre_x*pre_y*(pre_c+jump_c)))
    #TODO: finish this function
    return new_spike_per_neuron, new_n_spike_per_neuron
    

def add_padding(pre_spike_per_neuron, pre_n_spike_per_neuron, shape, padding):
    if pre_n_spike_per_neuron is None:
        b =''
    x, y, c = shape
    x_padd, y_padd = padding
    batch_size, spikes,max_n_spikes = pre_spike_per_neuron.shape

    if spikes != (x- x_padd)*(y- y_padd)*c:
        return pre_spike_per_neuron, pre_n_spike_per_neuron

    # put the array back into the original shape
    new_pre_spike_per_neuron = cp.reshape(pre_spike_per_neuron, (batch_size, (x- x_padd), (y- y_padd),c, max_n_spikes))

    padding_elements = [(0, 0)] + [(int(x_padd/2), int(x_padd/2))] + [(int(y_padd/2), int(y_padd/2))] + [(0, 0)] + [(0, 0)]  
    
    # Add padding to the new_pre_spike_per_neuron  
    padded_pre_spike_per_neuron = cp.pad(new_pre_spike_per_neuron, padding_elements, mode='constant',constant_values=cp.inf)
    
    padded_pre_spike_per_neuron = cp.reshape(padded_pre_spike_per_neuron, (batch_size, x*y*c, max_n_spikes))

    new_pre_n_spike_per_neuron = cp.reshape(pre_n_spike_per_neuron, (batch_size, (x- x_padd), (y- y_padd),c))
    padding_elements = [(0, 0)] + [(int(x_padd/2), int(x_padd/2))] + [(int(y_padd/2), int(y_padd/2))] + [(0, 0)]
    padded_pre_n_spike_per_neuron = cp.pad(new_pre_n_spike_per_neuron, padding_elements, mode='constant',constant_values= 0.)
    padded_pre_n_spike_per_neuron = cp.reshape(padded_pre_n_spike_per_neuron, (batch_size, x*y*c))
    # Now the n_spike_per_neuron
    # top_pad = cp.zeros((batch_size, x*int(y_padd/2)))
    # side_pads = cp.zeros((batch_size,int(x_padd/2)))
    # # we will start from the end to make it simpler
    # splits = cp.split(pre_n_spike_per_neuron, (x- x_padd), axis=1)
    # # top_pad = cp.zeros(splits[0].shape)
    # for i in range(len(splits)):
    #     splits[i] = cp.append(cp.append(side_pads, splits[i], axis= 1), side_pads, axis=1)
    # splits.insert(0, top_pad)
    # splits.append(top_pad)
    # new_pre_n_spike_per_neuron = cp.concatenate(splits, axis=1)
    return padded_pre_spike_per_neuron, padded_pre_n_spike_per_neuron



def trimed_errors(errors, previous_filter, channels):
    channels = int(channels)
    x_filter, y_filter, pre_channels = previous_filter
    padding_x_to_remove = int((x_filter-1)/2)
    padding_y_to_remove = int((y_filter-1)/2)
    batch_size, n_neurons, max_n_spike = errors.shape
    x = int(sqrt(n_neurons/channels))
    y = int(sqrt(n_neurons/channels))
    errors = cp.reshape(errors, (batch_size, x, y, channels, max_n_spike))
    errors = errors[:,padding_x_to_remove:-padding_x_to_remove,padding_y_to_remove:-padding_y_to_remove]
    new_n_neurons = (x-padding_x_to_remove*2) * (y-padding_y_to_remove*2) * channels
    shapped_errors = cp.reshape(errors, (batch_size, new_n_neurons, max_n_spike))
    return shapped_errors


# NOT USED
def add_padding_to_x_and_tau(x: cp.ndarray, tau: cp.ndarray, pre_shape: cp.ndarray, padding: list, padding_after: list, errors: cp.ndarray) -> cp.ndarray:
    raise RuntimeError("This function is not meant to be used")
    if errors.shape == x.shape:
        return x, tau
    error_batch, error_dim, channel = errors.shape
    x_dim, y_dim, c_dim = pre_shape
    x_padd, y_padd = padding
    x_padd_after, y_padd_after = padding_after
    batch_size, spikes, max_n_spikes = x.shape
    top_pad = cp.full((batch_size, (x_dim-x_padd + x_padd_after)*int(y_padd_after/2), max_n_spikes), cp.inf)
    side_pads = cp.full((batch_size,int(x_padd_after/2), max_n_spikes), cp.inf)
    # top_pad = cp.zeros((batch_size, (x_dim-x_padd + x_padd_after)*int(y_padd_after/2), max_n_spikes))
    # side_pads = cp.zeros((batch_size,int(x_padd_after/2), max_n_spikes))
    # we will start from the end to make it simpler
    splits_x = cp.split(x, (x_dim- x_padd), axis=1)
    splits_tau = cp.split(tau, (x_dim- x_padd), axis=1)
    for i in range(len(splits_x)):
        splits_x[i] = cp.append(cp.append(side_pads, splits_x[i], axis= 1), side_pads, axis=1)
        splits_tau[i] = cp.append(cp.append(side_pads, splits_tau[i], axis= 1), side_pads, axis=1)
    # if splits_x[0].shape !=top_pad.shape:
    #     raise RuntimeError("mismatch in padding dimensions, should not be possible")
    splits_x.insert(0, top_pad)
    splits_x.append(top_pad)
    splits_tau.insert(0, top_pad)
    splits_tau.append(top_pad)
    x_new_padded = cp.concatenate(splits_x, axis=1)
    tau_new_padded = cp.concatenate(splits_tau, axis=1)
    return x_new_padded, tau_new_padded