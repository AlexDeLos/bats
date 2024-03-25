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
            n = ''

    def backward(self, output_errors: cp.ndarray) \
            -> List[cp.array]:# type: ignore
        errors = output_errors
        gradient = []
        jump_layers = []
        errors_jump_array = []
        using_fuse = False
        for i, layer in enumerate(reversed(self.__layers)):
            

            if not layer.trainable:  # Reached input layer
                gradient.insert(0, None)
                break
            if layer in jump_layers:
                if jump_layers.count(layer) != 1:
                    #? maybe do something
                    qwe = ""
                index = jump_layers.index(layer) # this will give the first index of the layer, so the lowest layer
                #! errors might have different shapes
                # if this is connected to a res layer we run it two times
                if layer._is_residual:
                    weights_grad, (errors1,errors_jump1) = layer.backward(errors)
                    weights_grad_jump, (errors2,errors_jump2) = layer.backward(errors_jump_array[index])
                    #! we are adding the errors, is this good?
                    jump_layers.pop(index)
                    errors_jump_array.pop(index)
                    if errors1 is None:
                        errors = errors2
                    elif errors2 is None:
                        errors = errors1
                    else:
                        errors = errors1 + errors2
                    if errors_jump1 is None:
                        errors_jump = errors_jump2
                    elif errors_jump2 is None:
                        errors_jump = errors_jump1
                    else:
                        errors_jump = errors_jump1 + errors_jump2

                    if errors_jump is None:
                        jump_layer_is_input = True

                    errors_jump_array.append(errors_jump)
                    jump_layers.append(layer.jump_layer)# type: ignore

                else:
                    weights_grad_pre, errors1 = layer.backward(errors)
                    #! this is never used, how should we deal with the layer that recieves the jump?
                    #! error shape in conv can be wrong
                    if errors_jump_array[index] is not None:
                        weights_grad_jump, errors2 = layer.backward(errors_jump_array[index])
                    else:
                        errors2 = None
                    #! TESTING
                    weights_grad_jump = weights_grad_pre
                    jump_layers.pop(index)
                    errors_jump_array.pop(index)
                    if errors1 is None:
                        errors = errors2
                    elif errors2 is None:
                        errors = errors1
                    else:
                        errors = (errors1 + errors2)/2
                        # errors = errors1

                    if weights_grad_jump is None and weights_grad_pre is None:
                        weights_grad = None
                    else:
                        weights_grad = (weights_grad_jump + weights_grad_pre) / 2

            elif layer._is_residual:
                weights_grad, (errors,errors_jump) = layer.backward(errors)
                if errors_jump is None:
                    jump_layer_is_input = True

                errors_jump_array.append(errors_jump)
                jump_layers.append(layer.jump_layer)# type: ignore
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
        for layer, delta in zip(self.__layers, deltas):
            if delta is None:
                continue
            layer.add_deltas(delta)

    def store(self, dir_path: Path) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        for it in self.__layers:
            it.store(dir_path)

    def restore(self, dir_path: Path) -> None:
        for it in self.__layers:
            it.restore(dir_path)
