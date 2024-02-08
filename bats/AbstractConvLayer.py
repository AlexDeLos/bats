from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional
import cupy as cp

import numpy as np

from bats import AbstractLayer


class AbstractConvLayer(AbstractLayer):
    def __init__(self, neurons_shape: np.ndarray, name: str = "", use_padding: bool = False):
        if type(neurons_shape) == tuple:
            n_neurons = neurons_shape[0][0] * neurons_shape[0][1] * neurons_shape[0][2] + neurons_shape[1][0] * neurons_shape[1][1] * neurons_shape[1][2]
        else:
            n_neurons = neurons_shape[0] * neurons_shape[1] * neurons_shape[2]
        super(AbstractConvLayer, self).__init__(n_neurons, name)
        # if type(neurons_shape) == tuple:
        self._neurons_shape: cp.ndarray = cp.array(neurons_shape, dtype=cp.int32)
        self._use_padding: bool = use_padding

    @property
    def neurons_shape(self) -> cp.ndarray:
        return self._neurons_shape