from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cupy as cp

from bats import AbstractLayer
from bats.Layers import InputLayer


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
            self.__input_layer = layer

    def reset(self):
        for layer in self.__layers:
            layer.reset()

    def forward(self, spikes_per_neuron: np.ndarray, n_spikes_per_neuron: np.ndarray,
                max_simulation: float = np.inf, training: bool = False) -> None:
        self.__input_layer.set_spike_trains(spikes_per_neuron, n_spikes_per_neuron)
        for layer in self.__layers:
            layer.forward(max_simulation, training)

    def backward(self, output_errors: cp.ndarray) \
            -> List[cp.array]:
        errors = output_errors
        gradient = []
        for i, layer in enumerate(reversed(self.__layers)):
            if not layer.trainable:  # Reached input layer
                gradient.insert(0, None)
                break
            if layer._is_residual:
                # check for nans
                # check the shapes
                weights_grad, errors = layer.backward(errors)
                test = cp.any(cp.isnan(errors)).item() 
                #? why are there nans in the append case?
                #! Let's try forcing it to be zero
                errors = cp.nan_to_num(errors, nan=0.0)
                weights_grad = cp.nan_to_num(weights_grad, nan=0.0)
                qweqwe = 0
                # nan_mask = cp.isnan(errors)

                # Get the indices of NaN values
                # nan_indices = cp.where(nan_mask)
                # print(nan_indices)
                asdas= 1
            #problem here when the previous layer is a residual
            # errors.shape = (batch_size, n_neurons, max_n_spikes)
            # weights_grad.shape = (batch_size, pre_num_neurons , n_neurons)
            #BINGO: the problem is that when the layer is residual the pre_num_neurons is fucked up
            # when residual is used it should look like
            else:
                weights_grad, errors = layer.backward(errors)
                qweqwe = 0
                asdas= 1
            gradient.insert(0, weights_grad)
            #gradient can have different shapes
        return gradient

    def apply_deltas(self, deltas: List[cp.array]) -> None:
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
