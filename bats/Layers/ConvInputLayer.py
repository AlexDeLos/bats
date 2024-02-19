from re import split
from typing import Optional, Tuple
import numpy as np
import cupy as cp

from bats.AbstractConvLayer import AbstractConvLayer
from bats.AbstractLayer import AbstractLayer
from bats.Utils.utils import aped_on_channel_dim,split_on_channel_dim


class ConvInputLayer(AbstractConvLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.__indices: Optional[cp.ndarray] = None
        self.__times_per_neuron: Optional[cp.ndarray] = None
        self.__n_spike_per_neuron: Optional[cp.ndarray] = None

    @property
    def trainable(self) -> bool:
        return False

    @property
    def spike_trains(self) -> Tuple[cp.ndarray, cp.ndarray]:
        ret1 = self.__times_per_neuron
        # ret1_append,ret2_append = aped_on_channel_dim(self.__times_per_neuron, self.__n_spike_per_neuron, self.__times_per_neuron, self.__n_spike_per_neuron, self._neurons_shape, self._neurons_shape)
        ret2 = self.__n_spike_per_neuron
        # split1,split2 = split_on_channel_dim(ret2_append, self._neurons_shape)
        return ret1, ret2

    def set_spike_trains(self, times_per_neuron: np.ndarray, n_times_per_neuron: np.ndarray) -> None:
        # times_per_neuron = [[[0.01 , 0.001],[0.02, 0.002], [0.03, 0.003] ,[0.04, 0.004], [0.05, 0.005],
        #                      [0.11, 0.011], [0.12, 0.012], [0.13, 0.013], [0.14, 0.014], [0.15, 0.015],
        #                     [0.21, 0.021], [0.22, 0.022], [0.23, 0.023], [0.24, 0.024], [0.25, 0.025],
        #                     [0.31, 0.031], [0.32, 0.032], [0.33, 0.033], [0.34, 0.034], [0.35, 0.035],
        #                     [0.41, 0.041], [0.42, 0.042], [0.43, 0.043], [0.44, 0.044], [0.45, 0.045]],

        #                     [[0.01 , 0.001],[0.02, 0.002], [0.03, 0.003] ,[0.04, 0.004], [0.05, 0.005],
        #                      [0.11, 0.011], [0.12, 0.012], [0.13, 0.013], [0.14, 0.014], [0.15, 0.015],
        #                     [0.21, 0.021], [0.22, 0.022], [0.23, 0.023], [0.24, 0.024], [0.25, 0.025],
        #                     [0.31, 0.031], [0.32, 0.032], [0.33, 0.033], [0.34, 0.034], [0.35, 0.035],
        #                     [0.41, 0.041], [0.42, 0.042], [0.43, 0.043], [0.44, 0.044], [0.45, 0.045]],

        #                     [[0.01 , 0.001],[0.02, 0.002], [0.03, 0.003] ,[0.04, 0.004], [0.05, 0.005],
        #                      [0.11, 0.011], [0.12, 0.012], [0.13, 0.013], [0.14, 0.014], [0.15, 0.015],
        #                     [0.21, 0.021], [0.22, 0.022], [0.23, 0.023], [0.24, 0.024], [0.25, 0.025],
        #                     [0.31, 0.031], [0.32, 0.032], [0.33, 0.033], [0.34, 0.034], [0.35, 0.035],
        #                     [0.41, 0.041], [0.42, 0.042], [0.43, 0.043], [0.44, 0.044], [0.45, 0.045]]]
        
        # n_times_per_neuron = [[1]*25, [1]*25]
        # Old shapes = (2, 728 , 1) and (2, 728)
        # times_per_neuron = cp.zeros(times_per_neuron.shape, dtype=cp.float32)
        # n_times_per_neuron = cp.zeros(n_times_per_neuron.shape, dtype=cp.int32)
        self.__times_per_neuron = cp.array(times_per_neuron, dtype=cp.float32)
        self.__n_spike_per_neuron = cp.array(n_times_per_neuron, dtype=cp.int32)
        
    @property
    def weights(self) -> Optional[cp.ndarray]:
        return None

    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        pass
        
    def reset(self) -> None:
        pass

    def forward(self, max_simulation: float, training: bool = False) -> None:
        pass

    def backward(self, errors: cp.array, labels: Optional[cp.ndarray] = None) \
            -> Optional[Tuple[cp.ndarray, cp.ndarray]]:
        return None

    def add_deltas(self, delta_weights: cp.ndarray) -> None:
        pass