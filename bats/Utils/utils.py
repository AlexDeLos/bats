from typing import Tuple
import cupy as cp
from math import sqrt
import argparse
import os
# This file contains utility functions that are used in the main code
def get_arguments():
    parser = argparse.ArgumentParser(prog="SNN script",
                                     description="Run a Spiking Neural Network (SNN) on the given dataset")
    # parser.add_argument("gnn", choices=["GAT", "MessagePassing", "GraphSAGE", "GINE", "HeteroGNN"], default="GAT")
    # if file is moved in another directory level relative to the root (currently in root/utils/src), this needs to be changed
    root_directory = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    parser.add_argument("--cluster", default=False, type=bool)
    parser.add_argument("--fuse_func", default="Append", type=str)
    parser.add_argument("--residual_jump_length", default=2, type=int)
    parser.add_argument("--standard", default=False, type=bool)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--store", default=True, type=bool)
    parser.add_argument("--n_neurons", default=500, type=int)
    parser.add_argument("--alternate", default=False, type=bool)
    parser.add_argument("--residual_every_n", default=3, type=int)
    parser.add_argument("--n_hidden_layers", default=6, type=int)
    parser.add_argument("--use_residual",default=False, type=bool)
    parser.add_argument("--use_wanb", default=False, type=bool)
    parser.add_argument("--use_3_channels", default=True, type=bool)
    parser.add_argument("-s", "--save_model", action="store_true", default=False)
    parser.add_argument("-b", "--batch_size", default=20, type=int)
    parser.add_argument("-n", "--n_epochs", default=10, type=int)
    parser.add_argument("-l", "--learning_rate", default=None, type=float)
    parser.add_argument("--n_test_samples", default=10000, type=int)
    parser.add_argument("--n_train_samples", default=50000, type=int)
    parser.add_argument("--cifar100", default=False, type=bool)
    parser.add_argument("--batch_size_test", default=90, type=int)
    parser.add_argument("--use_coarse_labels", default=False, type=bool)
    parser.add_argument("--use_pad", default=True, type=bool)
    parser.add_argument("--use_delay", default=False, type=bool)
    parser.add_argument("--ttfs", default=False, type=bool)
    parser.add_argument("--restore", default=False, type=bool)


    args = parser.parse_args()
    return args


def split_errors_on_channel_dim(errors, shape):
    #! how are the channels saved in the errors?
    x, y, c = shape.get()
    # errors = cp.reshape(errors, (errors.shape[0], x, y, c, errors.shape[2]))
    errors = cp.reshape(errors, (errors.shape[0], x, y, c*2, errors.shape[2]))
    errors_pre, errors_jump = cp.split(errors, 2, axis=3)
    errors_pre = cp.reshape(errors_pre, (errors.shape[0], int(x*y*c), errors.shape[-1]))
    errors_jump = cp.reshape(errors_jump, (errors.shape[0], int(x*y*c), errors.shape[-1]))
    return errors_pre, errors_jump

def split_spike_per_neuron_on_channel_dim(spike_per_neuron, n_spike_per_neuron, shape):
    x, y, c = shape.get()
    spike_per_neuron = cp.reshape(spike_per_neuron, (spike_per_neuron.shape[0], x, y, c*2, spike_per_neuron.shape[2]))
    n_spike_per_neuron = cp.reshape(n_spike_per_neuron, (n_spike_per_neuron.shape[0], x, y, c*2))
    spike_per_neuron_pre, spike_per_neuron_jump = cp.split(spike_per_neuron, 2, axis=3)
    n_spike_per_neuron_pre, n_spike_per_neuron_jump = cp.split(n_spike_per_neuron, 2, axis=3)
    spike_per_neuron_pre = cp.reshape(spike_per_neuron_pre, (spike_per_neuron_pre.shape[0], int(x*y*c), spike_per_neuron_pre.shape[-1]))
    spike_per_neuron_jump = cp.reshape(spike_per_neuron_jump, (spike_per_neuron_jump.shape[0], int(x*y*c), spike_per_neuron_jump.shape[-1]))
    return spike_per_neuron_pre, n_spike_per_neuron_pre, spike_per_neuron_jump, n_spike_per_neuron_jump

def average_on_channel_dim(pre_spike_per_neuron, pre_n_spike_per_neuron, jump_spike_per_neuron, jump_n_spike_per_neuron, shape_of_neurons):
    batch_size, spikes, max_n_spikes = pre_spike_per_neuron.shape
    jump_batch_size, jump_spikes, jump_max_n_spikes = jump_spike_per_neuron.shape

    return None, None

def aped_on_channel_dim(pre_spike_per_neuron, pre_n_spike_per_neuron, jump_spike_per_neuron, jump_n_spike_per_neuron, shape_of_neurons, delay = False):
    batch_size, spikes, max_n_spikes = pre_spike_per_neuron.shape
    jump_batch_size, jump_spikes, jump_max_n_spikes = jump_spike_per_neuron.shape

    if delay:
        copy_pre_spike_per_neuron = cp.copy(pre_spike_per_neuron)
        non_inf_values_pre = copy_pre_spike_per_neuron[cp.isfinite(copy_pre_spike_per_neuron)]  # Select non-inf values
        average_non_inf_pre = cp.mean(non_inf_values_pre)
        copy_jump_spike_per_neuron = cp.copy(jump_spike_per_neuron)
        non_inf_values_jump = copy_jump_spike_per_neuron[cp.isfinite(copy_jump_spike_per_neuron)]
        average_non_inf_jump = cp.mean(non_inf_values_jump)
        time_delay = average_non_inf_pre - average_non_inf_jump
        jump_spike_per_neuron = jump_spike_per_neuron+ time_delay

    if batch_size != jump_batch_size:
        raise RuntimeError("The batch sizes of the two inputs are not the same")
    pre_x, pre_y, pre_c = shape_of_neurons.get()
    # jump_x, jump_y, jump_c = jump_shape.get()
    
    if max_n_spikes != jump_max_n_spikes:
        print('Warning: the maximum number of spikes is not the same for the two inputs', max_n_spikes, jump_max_n_spikes)
        max_n_spikes = max(max_n_spikes, jump_max_n_spikes)
        # now we make sure that the two inputs have the same number of spikes by adding 0s to the smaller one
        # TODO: does reshape do this?-> NO
        padding_for_max_spikes_pre = ([0,0],[0,0], [0,max_n_spikes-pre_spike_per_neuron.shape[2]])
        padding_for_max_spikes_jump = ([0,0],[0,0], [0,max_n_spikes-jump_spike_per_neuron.shape[2]])
        pre_spike_per_neuron = cp.pad(pre_spike_per_neuron, padding_for_max_spikes_pre, mode='constant', constant_values=cp.inf)
        jump_spike_per_neuron = cp.pad(jump_spike_per_neuron, padding_for_max_spikes_jump, mode='constant', constant_values=cp.inf)

    #? Am I using the right reshape order?
    pre_spike_per_neuron = cp.reshape(pre_spike_per_neuron, (batch_size,pre_x, pre_y, pre_c, max_n_spikes))
    jump_spike_per_neuron = cp.reshape(jump_spike_per_neuron, (jump_batch_size,pre_x, pre_y, pre_c, jump_max_n_spikes))
    pre_n_spike_per_neuron = cp.reshape(pre_n_spike_per_neuron, (batch_size,pre_x, pre_y, pre_c))
    jump_n_spike_per_neuron = cp.reshape(jump_n_spike_per_neuron, (jump_batch_size,pre_x, pre_y, pre_c))

    # Now we append pre_spike_per_neuron and jump_spike_per_neuron on the channel dimension
    new_spike_per_neuron = cp.append(pre_spike_per_neuron, jump_spike_per_neuron, axis=3)
    new_spike_per_neuron = cp.reshape(new_spike_per_neuron, (batch_size, pre_x*pre_y*(pre_c+pre_c), max_n_spikes))

    new_n_spike_per_neuron = cp.append(pre_n_spike_per_neuron, jump_n_spike_per_neuron, axis=3)
    new_n_spike_per_neuron = cp.reshape(new_n_spike_per_neuron, (batch_size, pre_x*pre_y*(pre_c+pre_c)))
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



def trimed_errors(errors, previous_filter, pre_channels):
    #! Future me, keep in mind these names do not really reflect the actual meaning of the variables
    _, x_filter, y_filter, _  = previous_filter
    channels = int(pre_channels)
    padding_x_to_remove = int((x_filter-1)/2)
    padding_y_to_remove = int((y_filter-1)/2)
    batch_size, n_neurons, max_n_spike = errors.shape
    x = int(sqrt(n_neurons/channels))
    if y_filter == 1:
        y = 1
        x = int(n_neurons/channels)
    else:
        y = int(sqrt(n_neurons/channels))
    errors = cp.reshape(errors, (batch_size, x, y, channels, max_n_spike))
    if padding_y_to_remove == 0:
        errors = errors[:,padding_x_to_remove:-padding_x_to_remove,:,:]
    else:
        errors = errors[:,padding_x_to_remove:-padding_x_to_remove,padding_y_to_remove:-padding_y_to_remove]
    new_n_neurons = (x-padding_x_to_remove*2) * (y-padding_y_to_remove*2) * channels
    shapped_errors = cp.reshape(errors, (batch_size, new_n_neurons, max_n_spike))
    return shapped_errors

def fuse_inputs_conv_avg(pre_input, pre_n_spike_per_neuron, jump_input, jump_n_spike_per_neuron, neurons_shape, delay) -> Tuple[cp.ndarray, cp.ndarray]:
    #TODO: add the delay
    if delay:
        copy_pre_spike_per_neuron = cp.copy(pre_input)
        non_inf_values_pre = copy_pre_spike_per_neuron[cp.isfinite(copy_pre_spike_per_neuron)]  # Select non-inf values
        average_non_inf_pre = cp.mean(non_inf_values_pre)
        copy_jump_spike_per_neuron = cp.copy(jump_input)
        non_inf_values_jump = copy_jump_spike_per_neuron[cp.isfinite(copy_jump_spike_per_neuron)]
        average_non_inf_jump = cp.mean(non_inf_values_jump)
        time_delay = average_non_inf_pre - average_non_inf_jump
        jump_input = jump_input+ time_delay
    
    result_count = cp.maximum(pre_n_spike_per_neuron, jump_n_spike_per_neuron)


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
    
    # return residual_input
    result_times = cp.where(xor_combined_mask,
                        get_non_infinite,
                      cp.mean(cp.array([ pre_input, pre_input ]), axis=0))

    return result_times, result_count