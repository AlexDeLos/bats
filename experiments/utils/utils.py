import json
import numpy as np
import cupy as cp
import wandb
from bats.Layers import InputLayer, LIFLayer, LIFLayerResidual
from bats.Losses import *
from bats.Layers.ConvInputLayer import ConvInputLayer
from bats.Layers.ConvLIFLayer import ConvLIFLayer
from bats.Layers.ConvLIFLayer_new_Residual import ConvLIFLayer_new_Residual

from bats.Layers.PoolingLayer import PoolingLayer
class wandb_handler:
    def __init__(self, project_name, experiment_name, config, cnn):
        self.project_name = project_name
        self.experiment_name = experiment_name
        if not cnn:
            self.config = {
                "Cluster": config['Cluster'],
                "Use_residual": config['Use_residual'],
                "N_HIDDEN_LAYERS": config['N_HIDDEN_LAYERS'],
                "residual_every_n": config['residual_every_n'],
                "residual_jump_length": config['residual_jump_length'],
                "n_of_train_samples": config['n_of_train_samples'],
                "n_of_test_samples": config['n_of_test_samples'],
                "learning_rate": config['learning_rate'],
                "batch_size": config['batch_size'],
                "dataset": config['dataset'],
                "epochs": config['epochs'],
                "Fuse_function": config['Fuse_function'],
                "neuron_var": config['neuron_var'],
                "neuron_out_var": config['neuron_out_var'],
                "neuron_res_var": config['neuron_res_var'],
                "true target": config['True_target'],
                "false target": config['False_target'],
                "use_delay": config['Use_delay'],
                "loss function": config['loss'],
                "architecture": "SNN",
                "version": "Testing delay 1.1.3",
            }
        else:
            self.config = {
                "Cluster": config['Cluster'],
                "Use_residual": config['Use_residual'],
                "Standard": config['Standard'],
                "N_HIDDEN_LAYERS": config['N_HIDDEN_LAYERS'],
                "batch_size": config['batch_size'],
                "residual_every_n": config['residual_every_n'],
                "residual_jump_length": config['residual_jump_length'],
                "use_padding": config['use_padding'],
                "n_of_train_samples": config['n_of_train_samples'],
                "n_of_test_samples": config['n_of_test_samples'],
                "conv_var": config['conv'],
                "conv_res_var": config['conv_res'],
                "learning_rate": config['learning_rate'],
                "architecture": "CNN",
                "dataset": config['dataset'],
                "epochs": config['epochs'],
                "true target": config['True_target'],
                "false target": config['False_target'],
                "loss function": config['loss'],
                "use_delay": config['Use_delay'],
                "version": "1.2.0",
            }
        self.run = wandb.init(project=project_name, name=experiment_name, config=self.config)
        # self.run = None
        self.cache = {}
    def save(self, log_dict):
        self.cache.update(log_dict)
    def log(self):
        for key,item in self.cache.items():
            if type(item) == np.ndarray and item.shape == (1,):
                self.cache[key] = float(item)
            if type(item) == cp.array and item.shape == (1,):
                self.cache[key] = float(item)
            else:
                try:
                    json.dumps(item)
                except:
                    self.cache[key] = float(item)
        # turn the cache into a JSON
        # cache_json = json.dumps(self.cache)
        self.run.log(self.cache)

    def finish(self):
        self.run.finish()



def build_network_SNN(network, weight_initializer,n_input, n_hidden, neuron_var,neuron_out_var ,res_neuron_var, use_residual, res_every_n, res_jump_l, fuse_function,use_delay):
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
                                            use_delay = use_delay,
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

def build_network_SCNN(network, weight_initializer_conv, weight_initializer_ff, input_shape, standard, n_hidden, conv_var, conv_res_var, fc_var, output_var, use_residual, res_every_n, res_jump_l, use_padding, use_delay):
    input_layer = ConvInputLayer(neurons_shape=input_shape, name="Input layer")
    network.add_layer(input_layer, input=True)
    if not standard:
        hidden_layers = []
        for i in range(n_hidden):
            if i == 0:
                conv = ConvLIFLayer(previous_layer=input_layer,
                                filters_shape=conv_var['filter'], use_padding=use_padding,
                                tau_s=conv_var['tau_s'],
                                filter_from_next=conv_var['filter'],
                                theta=conv_var['threshold_hat'],
                                delta_theta=conv_var['delta_threshold'],
                                weight_initializer=weight_initializer_conv,
                                max_n_spike=conv_var['spike_buffer_size'],
                                name="Convolution "+str(i))
            elif i % res_every_n == 0:
                if use_residual:
                    if i - res_jump_l < 0:
                        jump_layer = input_layer
                    else:
                        jump_layer = hidden_layers[i - res_jump_l]
                    conv = ConvLIFLayer_new_Residual(previous_layer=network.layers[-1], jump_layer=jump_layer, filters_shape=conv_res_var['filter'], use_padding=use_padding,
                                                     use_delay= use_delay,
                                tau_s=conv_res_var['tau_s'],
                                # filter_from_next=conv_res_var['filter'],
                                theta=conv_res_var['threshold_hat'],
                                delta_theta=conv_res_var['delta_threshold'],
                                weight_initializer=weight_initializer_conv,
                                max_n_spike=conv_res_var['spike_buffer_size'],
                                name="Convolution Residual "+str(i))
                else:
                    conv = ConvLIFLayer(previous_layer=network.layers[-1], filters_shape=conv_var['filter'], use_padding=use_padding,
                                tau_s=conv_var['tau_s'],
                                filter_from_next=conv_var['filter'],
                                theta=conv_var['threshold_hat'],
                                delta_theta=conv_var['delta_threshold'],
                                weight_initializer=weight_initializer_conv,
                                max_n_spike=conv_var['spike_buffer_size'],
                                name="Convolution "+str(i))
            else:
                conv = ConvLIFLayer(previous_layer=conv, filters_shape=conv_var['filter'], use_padding=use_padding,
                                tau_s=conv_var['tau_s'],
                                filter_from_next=conv_var['filter'],
                                theta=conv_var['threshold_hat'],
                                delta_theta=conv_var['delta_threshold'],
                                weight_initializer=weight_initializer_conv,
                                max_n_spike=conv_var['spike_buffer_size'],
                                name="Convolution "+str(i))
            hidden_layers.append(conv)
            network.add_layer(conv)
        

        pool_final = PoolingLayer(conv, name="Pooling final")
        network.add_layer(pool_final)

        feedforward = LIFLayer(previous_layer=pool_final, n_neurons=fc_var['n_neurons'], tau_s=fc_var['tau_s'],
                            theta=fc_var['threshold_hat'],
                            delta_theta=fc_var['delta_threshold'],
                            weight_initializer=weight_initializer_ff,
                            max_n_spike=fc_var['spike_buffer_size'],
                            name="Feedforward")
        network.add_layer(feedforward)

        output_layer = LIFLayer(previous_layer=feedforward, n_neurons=output_var['n_neurons'], tau_s=output_var['tau_s'],
                                theta=output_var['threshold_hat'],
                                delta_theta=output_var['delta_threshold'],
                                weight_initializer=weight_initializer_ff,
                                max_n_spike=output_var['spike_buffer_size'],
                                name="Output layer")
        network.add_layer(output_layer)
    #! end of standard network builder

    # pool_2 = PoolingLayer(conv, name="Pooling 2")
    # network.add_layer(pool_2)
    else:

        conv_1 = ConvLIFLayer(previous_layer=input_layer, filters_shape=conv_var['filter'], tau_s=conv_var['tau_s'],
                            use_padding=use_padding,
                            theta=conv_var['threshold_hat'],
                            delta_theta=conv_var['delta_threshold'],
                            weight_initializer=weight_initializer_conv,
                            max_n_spike=conv_var['spike_buffer_size'],
                            name="Convolution 1")
        network.add_layer(conv_1)

        # pool_1 = PoolingLayer(conv_1, name="Pooling 1")
        # network.add_layer(pool_1)

        conv_1_1 = ConvLIFLayer(previous_layer=conv_1, filters_shape=conv_var['filter'], tau_s=conv_var['tau_s'],
                            use_padding=use_padding,
                            theta=conv_var['threshold_hat'],
                            delta_theta=conv_var['delta_threshold'],
                            weight_initializer=weight_initializer_conv,
                            max_n_spike=conv_var['spike_buffer_size'],
                            name="Convolution 1.1")
        network.add_layer(conv_1_1)

        conv_1_5 = ConvLIFLayer_new_Residual(previous_layer=conv_1_1, jump_layer= conv_1,
                                filters_shape=conv_res_var['filter'], tau_s=conv_res_var['tau_s'],
                                use_padding=use_padding,
                                use_delay= use_delay,
                                theta=conv_res_var['threshold_hat'],
                                delta_theta=conv_res_var['delta_threshold'],
                                weight_initializer=weight_initializer_conv,
                                max_n_spike=conv_res_var['spike_buffer_size'],
                                name="Convolution-res 1.5")
        
        network.add_layer(conv_1_5)
        
        pool_1_5 = PoolingLayer(conv_1_5, name="Pooling 1.5")
        network.add_layer(pool_1_5)

        conv_2 = ConvLIFLayer(previous_layer=pool_1_5, filters_shape=conv_var['filter'], tau_s=conv_var['tau_s'],
                            use_padding=use_padding,
                            theta=conv_var['threshold_hat'],
                            delta_theta=conv_var['delta_threshold'],
                            weight_initializer=weight_initializer_conv,
                            max_n_spike=conv_var['spike_buffer_size'],
                            name="Convolution 2")
        network.add_layer(conv_2)

        pool_2 = PoolingLayer(conv_2, name="Pooling 2")
        network.add_layer(pool_2)

        feedforward = LIFLayer(previous_layer=pool_2, n_neurons=fc_var['n_neurons'], tau_s=fc_var['tau_s'],
                            theta=fc_var['threshold_hat'],
                            delta_theta=fc_var['delta_threshold'],
                            weight_initializer=weight_initializer_ff,
                            max_n_spike=fc_var['spike_buffer_size'],
                            name="Feedforward")
        network.add_layer(feedforward)

        output_layer = LIFLayer(previous_layer=feedforward, n_neurons=output_var['n_neurons'], tau_s=output_var['tau_s'],
                                theta=output_var['threshold_hat'],
                                delta_theta=output_var['delta_threshold'],
                                weight_initializer=weight_initializer_ff,
                                max_n_spike=output_var['spike_buffer_size'],
                                name="Output layer")
        network.add_layer(output_layer)
