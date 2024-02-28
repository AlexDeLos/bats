import zipfile
import shutil
import os
from download_file import download

if __name__ == "__main__":
    print("Downloading CIFAR100...")
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar100.npz'
    filename = 'cifar100.npz'
    
    download(url, filename)

    print("Done.")