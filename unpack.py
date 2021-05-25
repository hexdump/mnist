#!/usr/bin/env python3
# Copyright (C) 2021, Liam Schumm <contact@hexdump.email>.

import os
import h5py
import numpy as np
from scipy import ndimage

assert os.path.exists("train-images-idx3-ubyte")
assert os.path.exists("train-labels-idx1-ubyte")
assert os.path.exists("t10k-images-idx3-ubyte")
assert os.path.exists("t10k-labels-idx1-ubyte")

train_images = []
train_images_boolean = []
train_labels = []
test_images = []
test_images_boolean = []
test_labels = []

def read_32(f):
    return int.from_bytes(f.read(4), byteorder = "big")

def parse_idx3(f):
    magic = read_32(f)
    assert magic == 2051
    num_images = read_32(f)
    num_rows = read_32(f)
    num_columns = read_32(f)
    data = np.frombuffer(f.read(), dtype = np.ubyte)
    return data.reshape((num_images, num_rows, num_columns))

def parse_idx1(f):
    magic = read_32(f)
    assert magic == 2049
    num_items = read_32(f)
    data = np.frombuffer(f.read(), dtype = np.ubyte)
    # Since we're not reshaping (which would fail on an
    # invalid size), we'll just double check we have the
    # right number of items.
    assert data.shape == (num_items,)
    return data

def scale_images(images):
    assert len(images.shape) == 3
    return np.array([ndimage.zoom(image, 32 / 28) for image in images])

def threshold(images, cutoff):
    return (images.flatten() > cutoff).reshape(images.shape)

# Parse datasets from files, loading them into memory.
with open("train-images-idx3-ubyte", "rb") as f:
    train_images = scale_images(parse_idx3(f))
with open("train-labels-idx1-ubyte", "rb") as f:
    train_labels = parse_idx1(f)
with open("t10k-images-idx3-ubyte", "rb") as f:
    test_images = scale_images(parse_idx3(f))
with open("t10k-labels-idx1-ubyte", "rb") as f:
    test_labels = parse_idx1(f)

# Dump our "regular" MNIST (although it's scaled from the original
# 28x28).
with h5py.File("mnist-scaled.hdf5", "w") as f:
    f.create_dataset("train_images", data = train_images)
    f.create_dataset("train_labels", data = train_labels)
    f.create_dataset("test_images", data = test_images)
    f.create_dataset("test_labels", data = test_labels)

# Dump our boolean version of MNIST, thresholding values based on
# the average of all pixel values over the dataset.
cutoff = np.mean(np.append(train_images.flatten(),
                           test_images.flatten()))
with h5py.File("mnist-boolean.hdf5", "w") as f:
    f.create_dataset("train_images",
                     data = threshold(train_images, cutoff))
    f.create_dataset("train_labels",
                     data = train_labels)
    f.create_dataset("test_images",
                     data = threshold(test_images, cutoff))
    f.create_dataset("test_labels", data = test_labels)
