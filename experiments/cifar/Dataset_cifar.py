import os
import numpy as np
# import matplotlib.pyplot as plt

import warnings
from scipy.ndimage import rotate
warnings.filterwarnings('ignore')

from typing import Tuple
# from elasticdeform import deform_random_grid
import warnings
# import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, map_coordinates

warnings.filterwarnings("ignore")
import numpy as np

TIME_WINDOW = 100e-2 #! try changing this used to be 100e-3
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


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

class Dataset:
    def __init__(self, target_dir: str, use_multi_channel: bool = False, cifar100: bool = False, use_coarse_labels: bool = False):
        if cifar100:
            train = unpickle(os.path.join(target_dir, "train"))
            x_train = train[b'data']
            if use_coarse_labels:
                y_train = np.array(train[b'coarse_labels'])
            else:
                y_train = np.array(train[b'fine_labels'])

            test = unpickle(os.path.join(target_dir, "test"))
            x_test = test[b'data']
            if use_coarse_labels:
                y_test = np.array(test[b'coarse_labels'])
            else:
                y_test = np.array(test[b'fine_labels'])
        else:
            data_batch_1 = unpickle(os.path.join(target_dir, "data_batch_1"))
            data_batch_2 = unpickle(os.path.join(target_dir, "data_batch_2"))
            data_batch_3 = unpickle(os.path.join(target_dir, "data_batch_3"))
            data_batch_4 = unpickle(os.path.join(target_dir, "data_batch_4"))
            data_batch_5 = unpickle(os.path.join(target_dir, "data_batch_5"))
            x_train = data_batch_1[b'data']
            y_train = np.array(data_batch_1[b'labels'], dtype= 'uint8')
            test_batch = unpickle(os.path.join(target_dir, "test_batch"))
            x_train = np.concatenate([data_batch_1[b'data'], data_batch_2[b'data'], data_batch_3[b'data'], data_batch_4[b'data'], data_batch_5[b'data']])
            y_train = np.concatenate([data_batch_1[b'labels'], data_batch_2[b'labels'], data_batch_3[b'labels'], data_batch_4[b'labels'], data_batch_5[b'labels']])
            x_test = test_batch[b'data']
            y_test = np.array(test_batch[b'labels'], dtype= 'uint8')
        
        self.__use_multi_channel = use_multi_channel
        self.__train_X = x_train
        self.__train_labels = y_train
        self.__test_X = x_test
        self.__test_labels = y_test

    @property
    def train_labels(self) -> np.ndarray:
        return self.__train_labels

    @property
    def test_labels(self) -> np.ndarray:
        return self.__test_labels

    def __to_spikes(self, samples):
        images = samples.reshape((samples.shape[0], 3, 32, 32))

        # spike_times = samples.reshape((samples.shape[0], N_NEURONS, 3))# now this assumes 3 channels
        spike_times = TIME_WINDOW * (1 - (samples / MAX_VALUE))
        spike_times[spike_times == TIME_WINDOW] = np.inf
        stack = samples.reshape((samples.shape[0], 3,N_NEURONS))
        if self.__use_multi_channel:
            averaged_spike_times = stack
        else:
            # Here I take the mean of the 3 channels, it is also worth to try just running the 3 channels separately
            averaged_spike_times = np.mean(stack, axis=1, keepdims=True)
        spike_times_av = TIME_WINDOW * (1 - (averaged_spike_times / MAX_VALUE))
        spike_times_av[spike_times_av == TIME_WINDOW] = np.inf
        if self.__use_multi_channel:
            n_spike_per_neuron = np.isfinite(spike_times_av).astype('int').reshape((samples.shape[0], N_NEURONS*3))
            spike_times_av_ret = spike_times_av.reshape((samples.shape[0], N_NEURONS*3, 1))
        else:
            n_spike_per_neuron = np.isfinite(spike_times_av).astype('int').reshape((samples.shape[0], N_NEURONS))
            spike_times_av_ret = spike_times_av.reshape((samples.shape[0], N_NEURONS, 1))
        return spike_times_av_ret, n_spike_per_neuron

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
            #! this de-aligs the labets
            old = samples.copy()
            samples = samples.reshape((samples.shape[0], 3,32, 32))
            samples = samples.transpose(0,2,3,1)
            for image in range(samples.shape[0]):
                rotations = np.random.randint(0, high=4)
                samples[image] = rotate(samples[image], 90 * rotations,reshape=False)
                asasd = ''
            # undo the transpose
            samples = samples.transpose(0,3,1,2)
            sample_new = samples.reshape(old.shape)

            # samples = samples.reshape((samples.shape[0], N_NEURONS*3))
            # samples = deform_random_grid(list(samples), sigma=1.0, points=2, order=0) # this pakage is no on conda so can't be used
            samples = np.array(sample_new)
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
