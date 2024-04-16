from sympy import use
import wandb
from bats.Layers import InputLayer, LIFLayer, LIFLayerResidual
from bats.Losses import *
from bats.Network import Network
def build_network_SNN(network, weight_initializer,n_input, n_hidden, neuron_var,neuron_out_var ,res_neuron_var, use_residual, res_every_n, res_jump_l, fuse_function):
    input_layer = InputLayer(n_neurons=n_input, name="Input layer")
    network.add_layer(input_layer, input=True)
    hidden_layers = []
    for i in range(n_hidden):
        if i == 0:
            hidden_layer = LIFLayer(previous_layer=input_layer, n_neurons=neuron_var['n_neurons'], tau_s=neuron_var['tau_s'],
                                    theta=neuron_var['threshold_hat'],
                                    delta_theta=neuron_var['delta_threshold'],
                                    weight_initializer=weight_initializer,
                                    max_n_spike=neuron_var['spike_buffer_size'],
                                    name="Hidden layer 0")
        elif i % res_every_n ==0 and use_residual:
            if i - res_jump_l < 0:
                jump_layer = input_layer
            else:
                jump_layer = hidden_layers[i - res_jump_l]
            hidden_layer = LIFLayerResidual(previous_layer=hidden_layers[i-1],
                                            # jump_layer= hidden_layers[i-1],
                                            jump_layer = jump_layer,
                                            n_neurons=res_neuron_var['n_neurons'], tau_s=res_neuron_var['tau_s'],
                                            theta=res_neuron_var['threshold_hat'],
                                            fuse_function=fuse_function,
                                            delta_theta=res_neuron_var['delta_threshold'],
                                            weight_initializer=weight_initializer,
                                            max_n_spike=res_neuron_var['spike_buffer_size'],
                                            name="Residual layer " + str(i))
        elif i % res_every_n ==0 and not use_residual:
            hidden_layer = LIFLayer(previous_layer=hidden_layers[i-1], n_neurons=neuron_var['n_neurons'], tau_s=neuron_var['tau_s'],
                                    theta=neuron_var['threshold_hat'],
                                    delta_theta=neuron_var['delta_threshold'],
                                    weight_initializer=weight_initializer,
                                    max_n_spike=neuron_var['spike_buffer_size'],
                                    name="Hidden layer " + str(i))

        else:
            hidden_layer = LIFLayer(previous_layer=hidden_layers[i-1], n_neurons=neuron_var['n_neurons'], tau_s=neuron_var['tau_s'],
                                    theta=neuron_var['threshold_hat'],
                                    delta_theta=neuron_var['delta_threshold'],
                                    weight_initializer=weight_initializer,
                                    max_n_spike=neuron_var['spike_buffer_size'],
                                    name="Hidden layer " + str(i))
        hidden_layers.append(hidden_layer)
        network.add_layer(hidden_layer)
        
    output_layer = LIFLayer(previous_layer=hidden_layer, n_neurons=neuron_out_var['n_neurons'], tau_s=neuron_out_var['tau_s'], # type: ignore
                    theta=neuron_out_var['threshold_hat'],
                    delta_theta=neuron_out_var['delta_threshold'],
                    weight_initializer=weight_initializer,
                    max_n_spike=neuron_out_var['spike_buffer_size'],
                    name="Output layer")
    network.add_layer(output_layer)