from multiprocessing.spawn import old_main_modules
import os
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
        units = fileh.root.spikes.units
        times = fileh.root.spikes.times
        labels = fileh.root.labels

        print("Number of train samples: %d"%len(labels))
        spike_times = np.empty((len(times), NUMBER_OF_NEURONS), dtype=object)
        spike_counts = np.empty((len(times), NUMBER_OF_NEURONS), dtype=int)

        # Initialize each element as an empty list
        for i in range(len(times)):
            for j in range(NUMBER_OF_NEURONS):
                spike_times[i, j] = []

        # Processing loop
        for inputs in range(len(times)):
            print("Processing input %d" % inputs)
            for u, spike in enumerate(times[inputs]):
                spike_times[inputs, units[inputs][u]].append(spike)
                spike_counts[inputs, units[inputs][u]] += 1
        np.save("spike_times.npy", spike_times)
        spike_times = np.load("spike_times.npy",allow_pickle=True)
        new_spike_times = np.empty((len(times), NUMBER_OF_NEURONS,90), dtype=object)
        for i in range(len(spike_times)):
            print("Processing input %d" % i)
            for j in range(len(spike_times[i])):
                new_spike_times[i, j] = np.pad(np.array(spike_times[i, j]), (0, 90 - len(spike_times[i, j])), constant_values=np.inf)
                new_spike_times[i, j] = np.sort(new_spike_times[i, j])
        np.save("new_spike_times.npy", new_spike_times)
        print("Done new_spike_times")
        spike_counts = np.load("spike_counts.npy")
    
        print("Done")
        self.__train_spike_times = spike_times


        fn_test = files[1]
        origin = "%s/%s"%(base_url,fn_test)
        hdf5_file_path_test = get_and_gunzip(origin, fn_test, md5hash=file_hashes[fn_test])
        print(hdf5_file_path_test)
        print("Loading test data")
        fileh = tables.open_file(hdf5_file_path_train, mode='r')
        units = fileh.root.spikes.units
        times = fileh.root.spikes.times
        labels = fileh.root.labels
        print("Number of test samples: %d"%len(labels))
        spike_times = np.empty((len(times), NUMBER_OF_NEURONS), dtype=object)
        spike_counts = np.empty((len(times), NUMBER_OF_NEURONS), dtype=int)

        # Initialize each element as an empty list
        for i in range(len(times)):
            for j in range(NUMBER_OF_NEURONS):
                spike_times[i, j] = []

        # Processing loop
        for inputs in range(len(times)):
            print("Processing input %d" % inputs)
            for u, spike in enumerate(times[inputs]):
                spike_times[inputs, units[inputs][u]].append(spike)
        np.save("spike_times_test.npy", spike_times)
        spike_times = np.load("spike_times_test.npy",allow_pickle=True)
        new_spike_times = np.empty((len(times), NUMBER_OF_NEURONS,90), dtype=object)
        for i in range(len(spike_times)):
            print("Processing input %d" % i)
            for j in range(len(spike_times[i])):
                new_spike_times[i, j] = np.pad(np.array(spike_times[i, j]), (0, 90 - len(spike_times[i, j])), constant_values=np.inf)
                new_spike_times[i, j] = np.sort(new_spike_times[i, j])
        np.save("new_spike_times_test.npy", new_spike_times)
        print("Done new_spike_times")
        spike_counts = np.load("spike_counts_test.npy")
    
        print("Done")
