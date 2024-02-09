import cupy as cp
# This file contains utility functions that are used in the main code


def add_padding(pre_spike_per_neuron, pre_n_spike_per_neuron, shape, padding):
    x, y, c = shape
    batch_size, spikes,max_n_spikes = pre_spike_per_neuron.shape

    if spikes != x*y:
        RuntimeError("Mismatch in the number of spikes and the shape of the input")

    # put the array back into the original shape
    new_pre_spike_per_neuron = cp.reshape(pre_spike_per_neuron, (batch_size, x, y,c, max_n_spikes))

    padding_elements = [(0, 0)] + [(int(padding[0]/2), int(padding[0]/2))] + [(int(padding[1]/2), int(padding[1]/2))] + [(0, 0)] + [(0, 0)]  
    
    # Add padding to the new_pre_spike_per_neuron  
    padded_pre_spike_per_neuron = cp.pad(new_pre_spike_per_neuron, padding_elements, mode='constant',constant_values=cp.inf)
    
    padded_pre_spike_per_neuron = cp.reshape(padded_pre_spike_per_neuron, (batch_size, (x+padding[0])* (y+padding[1])*c, max_n_spikes))
    # Now the n_spike_per_neuron
    top_pad = cp.zeros((batch_size, (int(padding[0]/2)*2 +x)*int(padding[1]/2)))
    side_pads = cp.zeros((batch_size,int(padding[1]/2)))
    # we will start from the end to make it simpler
    splits = cp.split(pre_n_spike_per_neuron, x, axis=1)
    for i in range(len(splits)):
        splits[i] = cp.append(cp.append(side_pads, splits[i], axis= 1), side_pads, axis=1)
    if splits[0].shape !=top_pad.shape:
        RuntimeError("mismatch in padding dimensions, should not be possible")
    splits.insert(0, top_pad)
    splits.append(top_pad)
    new_pre_n_spike_per_neuron = cp.concatenate(splits, axis=1)
    return padded_pre_spike_per_neuron, new_pre_n_spike_per_neuron

def add_zeros_to_middle(arr, num_zeros, index):
    left_half = arr[:index]
    right_half = arr[index:]
    zeros = cp.zeros(num_zeros)
    return cp.concatenate((left_half, zeros, right_half))