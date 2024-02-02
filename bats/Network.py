from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cupy as cp

from bats import AbstractLayer
from bats.Layers import InputLayer, PoolingLayer


class Network:
    def __init__(self):
        self.__layers: List[AbstractLayer] = []
        self.__input_layer: Optional[InputLayer] = None

    @property
    def layers(self) -> List[AbstractLayer]:
        return self.__layers

    @property
    def output_spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        return self.__layers[-1].spike_trains

    def add_layer(self, layer: AbstractLayer, input: bool = False) -> None:
        self.__layers.append(layer)
        if input:
            self.__input_layer = layer # type: ignore

    def reset(self):
        for layer in self.__layers:
            layer.reset()

    def forward(self, spikes_per_neuron: np.ndarray, n_spikes_per_neuron: np.ndarray,
                max_simulation: float = np.inf, training: bool = False) -> None:
        self.__input_layer.set_spike_trains(spikes_per_neuron, n_spikes_per_neuron)# type: ignore
        for layer in self.__layers:
            layer.forward(max_simulation, training)

    def backward(self, output_errors: cp.ndarray) \
            -> List[cp.array]:# type: ignore
        errors = output_errors
        gradient = []
        jump_layer = None
        using_fuse = False
        for i, layer in enumerate(reversed(self.__layers)):
            

            if not layer.trainable:  # Reached input layer
                gradient.insert(0, None)
                break
            if layer == jump_layer:
                #! errors might have different shapes
                if using_fuse:
                    # if this is connected to a res layer we run it two times
                    #! this is never used, how should we deal with the layer that recieves the jump?
                    weights_grad_jump, errors_jump = layer.backward(errors_jump)
                    weights_grad_pre, errors = layer.backward(errors)
                    if weights_grad_jump is None and weights_grad_pre is None:
                        weights_grad = None
                    else:
                        weights_grad = (weights_grad_jump + weights_grad_pre) / 2
                    # weights_grad = (weights_grad, weights_grad_jump)
                else:
                    weights_grad, errors = layer.backward(errors) # type: ignore
            elif layer._is_residual:
                if layer.fuse_function == "Append":
                    using_fuse = True
                    weights_grad, (errors,errors_jump) = layer.backward(errors)
                    if errors_jump is None:
                        jump_layer_is_input = True
                    jump_layer = layer.jump_layer# type: ignore
                else:
                    weights_grad, errors = layer.backward(errors)# type: ignore
                    jump_layer = layer.jump_layer# type: ignore
            else:
                #? how on earth is the gradient flowing in the residual?
                weights_grad, errors = layer.backward(errors) # type: ignore
                # TODO: Look into this thing
                #? how does back propagation work with residual?
                # It updates the res, and the res receives an update from the input (always have errors), but 
                # the jump layer should also learn, but it doesn't
            gradient.insert(0, weights_grad)
            #gradient can have different shapes
        return gradient

    def apply_deltas(self, deltas: List[cp.array]) -> None:# type: ignore
        for layer, deltas in zip(self.__layers, deltas):
            if deltas is None:
                continue
            layer.add_deltas(deltas)

    def store(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        for it in self.__layers:
            it.store(dir_path)

    def restore(self, dir_path: Path) -> None:
        for it in self.__layers:
            it.restore(dir_path)
