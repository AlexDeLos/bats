import tensorflow as tf 
from tensorflow import keras 

import numpy as np

import warnings 
warnings.filterwarnings('ignore')

from pathlib import Path
from typing import Tuple
# from elasticdeform import deform_random_grid
import warnings
# import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

warnings.filterwarnings("ignore")

from tensorflow import keras
import numpy as np

TIME_WINDOW = 100e-3
MAX_VALUE = 255
RESOLUTION = 32
N_NEURONS = RESOLUTION * RESOLUTION
ELASTIC_ALPHA_RANGE = [8, 10]
ELASTIC_SIGMA = 3
WIDTH_SHIFT = 0
HEIGHT_SHIFT = 0
ZOOM_RANGE = 12 / 100
ROTATION_RANGE = 12




def elastic_transform(image, alpha_range, sigma):
    if np.isscalar(alpha_range):
        alpha = alpha_range
    else:
        alpha = np.random.uniform(low=alpha_range[0], high=alpha_range[1])

    shape = image.shape
    dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=0, mode='reflect').reshape(shape)


class Dataset:
    def __init__(self, path: Path, use_multi_channel: bool = False, cifar100: bool = False):
        if cifar100:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        else:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        self.__use_multi_channel = use_multi_channel
        self.__train_X = x_train
        self.__train_labels = y_train
        self.__test_X = x_test
        self.__test_labels = y_test
        self.__datagen = keras.preprocessing.image.ImageDataGenerator(width_shift_range=WIDTH_SHIFT,
                                                                      height_shift_range=HEIGHT_SHIFT,
                                                                      zoom_range=ZOOM_RANGE,
                                                                      rotation_range=ROTATION_RANGE,
                                                                      fill_mode="nearest",
                                                                      preprocessing_function=lambda x:
                                                                      elastic_transform(x,
                                                                                        alpha_range=ELASTIC_ALPHA_RANGE,
                                                                                        sigma=ELASTIC_SIGMA))

    @property
    def train_labels(self) -> np.ndarray:
        return self.__train_labels

    @property
    def test_labels(self) -> np.ndarray:
        return self.__test_labels

    def __to_spikes(self, samples):
        spike_times = samples.reshape((samples.shape[0], N_NEURONS, 3))# now this assumes 3 channels
        spike_times = TIME_WINDOW * (1 - (spike_times / MAX_VALUE))
        spike_times[spike_times == TIME_WINDOW] = np.inf
        if self.__use_multi_channel:
            spike_times = spike_times.reshape((samples.shape[0], N_NEURONS*3, 1))
            n_spike_per_neuron_3_channels = np.isfinite(spike_times).astype('int').reshape((samples.shape[0], N_NEURONS*3))
            return spike_times, n_spike_per_neuron_3_channels

        else:
            avaraged_spike_times = np.mean(spike_times, axis=2, keepdims=True) # Here I take the mean of the 3 channels, it is also worth to try just running the 3 channels separately
            n_spike_per_neuron = np.isfinite(avaraged_spike_times).astype('int').reshape((samples.shape[0], N_NEURONS))
            return avaraged_spike_times, n_spike_per_neuron

    def shuffle(self) -> None:
        shuffled_indices = np.arange(len(self.__train_labels))
        np.random.shuffle(shuffled_indices)
        self.__train_X = self.__train_X[shuffled_indices]
        self.__train_labels = self.__train_labels[shuffled_indices]

    def __get_batch(self, samples, labels, batch_index, batch_size, augment) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = batch_index * batch_size
        end = start + batch_size

        samples = samples[start:end]
        labels = labels[start:end]

        if augment:
            """plt.imshow(samples[0])
            plt.show()"""
            """rotate = np.random.uniform(-ROTATION_RANGE, ROTATION_RANGE)
            zoom = np.random.uniform(1.0 - ZOOM_RANGE, 1.0 + ZOOM_RANGE)"""
            # samples = deform_random_grid(list(samples), sigma=1.0, points=2, order=0) #!add it back later IDK why it fails in the cluster
            samples = np.array(samples)
            """samples = np.expand_dims(samples, axis=3)
            samples = self.__datagen.flow(samples, batch_size=len(samples), shuffle=False).next()
            samples = samples[..., 0]"""
            """plt.imshow(samples[0])
            plt.show()
            exit()"""

        spikes_per_neuron, n_spikes_per_neuron = self.__to_spikes(samples)
        return spikes_per_neuron, n_spikes_per_neuron, labels

    def get_train_batch(self, batch_index: int, batch_size: int, augment: bool = False) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__train_X, self.__train_labels,
                                batch_index, batch_size, augment)

    def get_test_batch(self, batch_index: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__test_X, self.__test_labels,
                                batch_index, batch_size, False)

    def get_test_image_at_index(self, index):
        return self.__test_X[index]
