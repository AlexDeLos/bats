import re
import cupy as cp
from math import sqrt
# This file contains utility functions that are used in the main code


def add_padding(pre_spike_per_neuron, pre_n_spike_per_neuron, shape, padding):
    if pre_n_spike_per_neuron is None:
        b =''
    x, y, c = shape
    x_padd, y_padd = padding
    batch_size, spikes,max_n_spikes = pre_spike_per_neuron.shape

    if spikes != (x- x_padd)*(y- y_padd)*c:
        return pre_spike_per_neuron, pre_n_spike_per_neuron
        #! why is it only conv 1 that needs the padding?
        #? HOW IS THEIR PADDING ADDED?
        raise RuntimeError("Mismatch in the number of spikes and the shape of the input")

    # put the array back into the original shape
    new_pre_spike_per_neuron = cp.reshape(pre_spike_per_neuron, (batch_size, (x- x_padd), (y- y_padd),c, max_n_spikes))

    padding_elements = [(0, 0)] + [(int(x_padd/2), int(x_padd/2))] + [(int(y_padd/2), int(y_padd/2))] + [(0, 0)] + [(0, 0)]  
    
    # Add padding to the new_pre_spike_per_neuron  
    padded_pre_spike_per_neuron = cp.pad(new_pre_spike_per_neuron, padding_elements, mode='constant',constant_values=cp.inf)
    
    padded_pre_spike_per_neuron = cp.reshape(padded_pre_spike_per_neuron, (batch_size, x*y*c, max_n_spikes))
    # Now the n_spike_per_neuron
    top_pad = cp.zeros((batch_size, x*int(y_padd/2)))
    side_pads = cp.zeros((batch_size,int(x_padd/2)))
    # we will start from the end to make it simpler
    splits = cp.split(pre_n_spike_per_neuron, (x- x_padd), axis=1)
    # top_pad = cp.zeros(splits[0].shape)
    for i in range(len(splits)):
        splits[i] = cp.append(cp.append(side_pads, splits[i], axis= 1), side_pads, axis=1)
    # if splits[0].shape != top_pad.shape:
    #     raise RuntimeError("mismatch in padding dimensions, should not be possible")
    splits.insert(0, top_pad)
    splits.append(top_pad)
    new_pre_n_spike_per_neuron = cp.concatenate(splits, axis=1)
    return padded_pre_spike_per_neuron, new_pre_n_spike_per_neuron

def add_padding_to_x_and_tau(x: cp.ndarray, tau: cp.ndarray, pre_shape: cp.ndarray, padding: list, padding_after: list, errors: cp.ndarray) -> cp.ndarray:
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

def trimed_errors(errors, previous_filter, channels):
    channels = int(channels)
    x_filter, y_filter, _ = previous_filter
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