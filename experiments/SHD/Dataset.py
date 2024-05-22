import os
from typing import Tuple
import urllib.request
import gzip, shutil
from tensorflow.keras.utils import get_file
import tables
import numpy as np
import pickle

NUMBER_OF_NEURONS = 700
cache_dir = os.path.expanduser("./datasets/data")
cache_subdir="hdspikes"

def save_spike_times(spike_times, filename):
    with open(filename, 'wb') as f:
        pickle.dump(spike_times, f)


class Dataset:
    def __init__(self, path: str, n_train_samples: int = 8332, n_test_samples: int = 2088, bias=False):
        try:
            self.__train_spike_times = np.load("new_spike_times_train.npy",allow_pickle=True)
            self.__test_spike_times = np.load("new_spike_times_test.npy",allow_pickle=True)
            self.__train_spike_counts = np.load("spike_counts_train.npy")
            self.__test_spike_counts = np.load("spike_counts_test.npy")
            self.__train_labels = np.load("labels_train.npy")
            self.__test_labels = np.load("labels_test.npy")
        except FileNotFoundError:
            print("Downloading SHD dataset")
            print("Using cache dir: %s"%cache_dir)
            # The remote directory with the data files
            base_url = "https://zenkelab.org/datasets"
            # Retrieve MD5 hashes from remote
            response = urllib.request.urlopen("%s/md5sums.txt"%base_url)
            data = response.read() 
            lines = data.decode('utf-8').split("\n")
            file_hashes = { line.split()[1]:line.split()[0] for line in lines if len(line.split())==2 }
            def get_and_gunzip(origin, filename, md5hash=None):
                gz_file_path = get_file(filename, origin, md5_hash=md5hash, cache_dir=cache_dir, cache_subdir=cache_subdir)
                hdf5_file_path=gz_file_path[:-3]
                if not os.path.isfile(hdf5_file_path) or os.path.getctime(gz_file_path) > os.path.getctime(hdf5_file_path):
                    print("Decompressing %s"%gz_file_path)
                    with gzip.open(gz_file_path, 'r') as f_in, open(hdf5_file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                return hdf5_file_path
            # Download the Spiking Heidelberg Digits (SHD) dataset
            files = [ "shd_train.h5.gz", 
                    "shd_test.h5.gz",
                    ]
            fn_train = files[0]
            origin = "%s/%s"%(base_url,fn_train)
            hdf5_file_path_train = get_and_gunzip(origin, fn_train, md5hash=file_hashes[fn_train])
            print(hdf5_file_path_train)
            

            fileh = tables.open_file(hdf5_file_path_train, mode='r')
            units_train = fileh.root.spikes.units
            times_train = fileh.root.spikes.times
            labels_train = fileh.root.labels

            np.save("labels_train.npy", np.array(labels_train))

            print("Number of train samples: %d"%len(labels_train))
            spike_times_train = np.empty((len(times_train), NUMBER_OF_NEURONS), dtype=object)
            spike_counts_train = np.empty((len(times_train), NUMBER_OF_NEURONS), dtype=int)

            # Initialize each element as an empty list
            for i in range(len(times_train)):
                for j in range(NUMBER_OF_NEURONS):
                    spike_times_train[i, j] = []

            # Processing loop
            for inputs in range(len(times_train)):
                print("Processing input %d" % inputs)
                for u, spike in enumerate(times_train[inputs]):
                    spike_times_train[inputs, units_train[inputs][u]].append(spike)
                    spike_counts_train[inputs, units_train[inputs][u]] += 1
            np.save("spike_times_train.npy", spike_times_train)
            np.save("spike_counts_train.npy", spike_counts_train)
            print("Done new_spike_times")
            spike_counts_train = np.load("spike_counts_train.npy")
        
            print("Done train")
            self.__train_spike_times = spike_times_train


            fn_test = files[1]
            origin = "%s/%s"%(base_url,fn_test)
            hdf5_file_path_test = get_and_gunzip(origin, fn_test, md5hash=file_hashes[fn_test])
            print(hdf5_file_path_test)
            print("Loading test data")
            fileh = tables.open_file(hdf5_file_path_test, mode='r')
            units_test = fileh.root.spikes.units
            times_test = fileh.root.spikes.times
            labels_test = fileh.root.labels

            np.save("labels_test.npy", np.array(labels_test))

            print("Number of test samples: %d"%len(labels_test))
            spike_times_test = np.empty((len(times_test), NUMBER_OF_NEURONS), dtype=object)
            spike_counts_test = np.empty((len(times_test), NUMBER_OF_NEURONS), dtype=int)

            # Initialize each element as an empty list
            for i in range(len(times_test)):
                for j in range(NUMBER_OF_NEURONS):
                    spike_times_test[i, j] = []

            # Processing loop
            for inputs in range(len(times_test)):
                print("Processing input test %d" % inputs)
                for u, spike in enumerate(times_test[inputs]):
                    spike_times_test[inputs, units_test[inputs][u]].append(spike)
                    spike_counts_test[inputs, units_test[inputs][u]] += 1
            np.save("spike_times_test.npy", spike_times_test)
            np.save("spike_counts_test.npy", spike_counts_test)

            max_len = max(spike_counts_train.max(), spike_counts_test.max())

            spike_times_train = np.load("spike_times_train.npy",allow_pickle=True)
            new_spike_times_train = np.empty((len(times_train), NUMBER_OF_NEURONS,max_len), dtype=float)
            for i in range(len(spike_times_train)):
                print("Processing input train 2 %d" % i)
                for j in range(len(spike_times_train[i])):
                    new_spike_times_train[i, j] = np.pad(np.array(spike_times_train[i, j]), (0, max_len - len(spike_times_train[i, j])), constant_values=np.inf)
                    new_spike_times_train[i, j] = np.sort(new_spike_times_train[i, j])
            np.save("new_spike_times_train.npy", new_spike_times_train)


            spike_times_test = np.load("spike_times_test.npy",allow_pickle=True)
            new_spike_times_test = np.empty((len(times_test), NUMBER_OF_NEURONS,max_len), dtype=float)
            for i in range(len(spike_times_test)):
                print("Processing input test 2 %d" % i)
                for j in range(len(spike_times_test[i])):
                    new_spike_times_test[i, j] = np.pad(np.array(spike_times_test[i, j]), (0, max_len - len(spike_times_test[i, j])), constant_values=np.inf)
                    new_spike_times_test[i, j] = np.sort(new_spike_times_test[i, j])
            np.save("new_spike_times_test.npy", new_spike_times_test)
            print("Done new_spike_times")
        
        print("Done")
    @property
    def train_labels(self) -> np.ndarray:
        return self.__train_labels

    @property
    def test_labels(self) -> np.ndarray:
        return self.__test_labels
    
    def shuffle(self) -> None:
        shuffled_indices = np.arange(len(self.__train_labels))
        np.random.shuffle(shuffled_indices)
        self.__train_spike_times = self.__train_spike_times[shuffled_indices]
        self.__train_labels = self.__train_labels[shuffled_indices]

    def __get_batch(self, samples_spikes, samples_count, labels, batch_index, batch_size, augment) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        start = batch_index * batch_size
        end = start + batch_size

        samples_spikes = samples_spikes[start:end]
        samples_count = samples_count[start:end]
        labels = labels[start:end]


        spikes_per_neuron = samples_spikes
        n_spikes_per_neuron = samples_count
        return spikes_per_neuron, n_spikes_per_neuron, labels

    def get_train_batch(self, batch_index: int, batch_size: int, augment: bool = False) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__train_spike_times, self.__train_spike_counts, self.__train_labels,
                                batch_index, batch_size, augment)

    def get_test_batch(self, batch_index: int, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.__get_batch(self.__test_spike_times, self.__test_spike_counts, self.__test_labels,
                                batch_index, batch_size, False)

    def get_test_image_at_index(self, index):
        return self.__test_spike_times[index]