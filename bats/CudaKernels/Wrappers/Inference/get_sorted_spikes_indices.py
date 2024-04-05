from math import exp
import cupy as cp
import numpy as np

"""
Get indices of spikes sorted in time in an array of spike times per neuron.
"""
def get_sorted_spikes_indices(spike_times_per_neuron, n_spike_per_neuron):
    batch_size, n_neurons, max_n_spike = spike_times_per_neuron.shape
    new_shape = (batch_size, n_neurons * max_n_spike)
    # Adds all spikes to the same dimension
    spike_times_reshaped = cp.reshape(spike_times_per_neuron, new_shape)

    # total_spikes => number of spikes per neuron
    total_spikes = cp.sum(n_spike_per_neuron, axis=1)
    #max spikes a neuron got
    #memory error here
    try:
        max_total_spike = int(cp.max(total_spikes))
    except ValueError:
        print('total_spikes:', total_spikes)
        print('n_spike_per_neuron:', n_spike_per_neuron)
        print('spike_times_per_neuron:', spike_times_per_neuron)
        print('spike_times_reshaped:', spike_times_reshaped)
        print('new_shape:', new_shape)
        print('batch_size:', batch_size)
        print('n_neurons:', n_neurons)
        print('max_n_spike:', max_n_spike)
        print('total_spikes:', total_spikes)
        print('total_spikes:', total_spikes)
        raise ValueError('Triggered error')
    """sorted_indices = cp.argsort(spike_times_reshaped, axis=1)[:, :max_total_spike]"""
    # creates keys? (was orriginally just called n)
    keys = np.arange(max_total_spike)
    # keys produces: ValueError: kth(=7200) out of bounds 7200
    sorted_indices = cp.argpartition(spike_times_reshaped, keys, axis=1)[:, :max_total_spike]
    #? what if I add padding to the indeces? increase theyr number to what they should be?
    return new_shape, sorted_indices, spike_times_reshaped