import numpy as np
import pickle
import gzip
import os
from torchvision import datasets

def create_pkl_gz():
    os.makedirs("mnist", exist_ok=True)

    print("Loading MNIST data...")
    train_dataset = datasets.MNIST(root='./mnist_data', train=True, download=True)
    test_dataset = datasets.MNIST(root='./mnist_data', train=False, download=True)
    train_images = train_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    train_labels = train_dataset.targets.numpy().astype(np.int32)
    test_images = test_dataset.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
    test_labels = test_dataset.targets.numpy().astype(np.int32)

    with gzip.open("mnist/training.pkl.gz", 'wb') as f:
        pickle.dump((train_images, train_labels), f)
    with gzip.open("mnist/testing.pkl.gz", 'wb') as f:
        pickle.dump((test_images, test_labels), f)
    print("Created mnist/training.pkl.gz and mnist/testing.pkl.gz")

if __name__ == "__main__":
    try:
        import torchvision
    except ImportError:
        print("Installing torchvision...")
        os.system("pip install torchvision")
    create_pkl_gz()
