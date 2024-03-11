import os
import pickle
from download_file import download
import tarfile

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
def extract_cifar10(tar_filename, target_dir):
    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Open the tar file
    with tarfile.open(tar_filename, 'r') as tar:
        # Extract each file
        for member in tar.getmembers():
            if member.name == 'cifar-100-python/train' or member.name == 'cifar-100-python/test':
                tar.extract(member, target_dir)
                print(f"Extracted: {member.name}")


if __name__ == "__main__":
    print("Downloading CIFAR100...")
    url = 'https://www.cs.toronto.edu/%7Ekriz/cifar-100-python.tar.gz'
    filename = 'cifar100.tar.gz'
    
    download(url, filename)

    target_dir = "./datasets/"
    print("Extracting CIFAR100...")
    extract_cifar10(filename, target_dir)
    #delete the tar file
    os.remove(filename)

    
    print("Done.")