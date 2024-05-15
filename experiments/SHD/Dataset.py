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
        # times = np.array([np.pad(t, (0, max_len-len(t)), 'constant') for t in times])
        spike_times = [[[]]*NUMBER_OF_NEURONS]*len(times)
        # spike_counts = np.zeros((len(times), NUMBER_OF_NEURONS))
        for inputs in range(len(times)):
            for u,spike in enumerate(times[inputs]):
                spike_times[inputs][units[inputs][u]].append(spike) #0,384 / 0,680 0, 465
                # spike_counts[inputs][units[inputs][u]] += 1
        # np.save("spike_counts.npy", spike_counts)
        spike_times_np = np.array(spike_times)
        np.save("spike_times.npy", spike_times_np)
        self.__train_spike_times = spike_times_np


        fn_test = files[1]
        origin = "%s/%s"%(base_url,fn_test)
        hdf5_file_path_test = get_and_gunzip(origin, fn_test, md5hash=file_hashes[fn_test])
        print(hdf5_file_path_test)
        fileh = tables.open_file(hdf5_file_path_train, mode='r')
        units = fileh.root.spikes.units
        times = fileh.root.spikes.times
        labels = fileh.root.labels
        print("Number of test samples: %d"%len(labels))
