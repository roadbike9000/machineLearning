# Load MNIST images into two matrices, one for training and, one for testing
import numpy as np
import gzip
import struct
import os
BASE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.dirname(BASE_DIR)


def load_images(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Read the header information into a bunch of variables
        _ignored, n_images, columns, rows = struct.unpack('>IIII', f.read(16))
        # Read all the pixels into a NumPy array of bytes:
        all_pixels = np.frombuffer(f.read(), dtype=np.uint8)
        # Reshape the pixels into a matrix where each line is an image:
        return all_pixels.reshape(n_images, columns * rows)


def prepend_bias(X):
    # Insert a column of 1s in the position 0 of X.
    # ("axis=1" stands form: "insert a column, not a row")
    return np.insert(X, 0, 1, axis=1)


# 60,000 images, each 785 elements (1 bias + 28 * 28 pixels)
X_TRAIN_FILENAME = os.path.join(PARENT_DIR, 'data', 'mnist\train-images-idx3-ubyte.gz')
X_train = prepend_bias(load_images(X_TRAIN_FILENAME))

# 10,000 images each 785 elements, with the same structure as X_train
X_TEST_FILENAME = os.path.join(PARENT_DIR, 'data', 'mnist\t10k-images-idx3-ubyte.gz')
X_test = prepend_bias(load_images(X_TEST_FILENAME))


# load and prepare MNIST labels
def load_labels(filename):
    # Open and unzip the file of images:
    with gzip.open(filename, 'rb') as f:
        # Skip the header bytes:
        f.read(8)
        # Read all lthe labels in to a list:
        all_labels = f.read()
        # Reshape the list of labels into a one-column matrix
        return np.frombuffer(all_labels, dtype=np.uint8).reshape(-1, 1)


def encode_fives(Y):
    # Convert all 5s to 1, and everything else to 0
    return (Y == 5).astype(int)


# 60k labels, each with value of 1 if the digit is a 5 and 0 otherwise
Y_TRAIN_FILENAME = os.path.join(PARENT_DIR, 'data', 'mnist\train-labels-idx-ubyte.gz')
Y_train = encode_fives(load_labels(Y_TRAIN_FILENAME))

# 10k labels, with the same encoding as Y_train
Y_TEST_FILENAME = os.path.join(PARENT_DIR, 'data', 'mnist\t10k-labels-idx-ubyte.gz')
Y_test = encode_fives(load_labels(Y_TEST_FILENAME))
