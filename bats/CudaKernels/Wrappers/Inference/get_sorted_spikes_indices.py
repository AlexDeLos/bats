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
    #! Error found here: 1 (found here when running the program on the cluster)
    #!TODO: Here is where you start next time bro, GL <3
    max_total_spike = int(cp.max(total_spikes))
    sorted_indices = cp.argsort(spike_times_reshaped, axis=1)[:, :max_total_spike]
    # creates keys? (was orriginally just called n)
    keys = np.arange(max_total_spike)
    # sorted_indices = cp.argpartition(spike_times_reshaped, keys, axis=1)[:, :max_total_spike] #!This was what "works"
    return new_shape, sorted_indices, spike_times_reshaped