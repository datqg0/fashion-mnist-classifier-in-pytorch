"""
Fashion-MNIST Classifier in PyTorch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_fashion_mnist
import os
import gzip
import tempfile
import urllib.request

import numpy as np
import torch
def load_fashion_mnist(n_train=10000, n_test=2000):
    # TODO: download the four idx gz files once, parse with np.frombuffer, return float tensors in 0-1 and int64 labels.
    pass
    base_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"

    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }
    temp_dir = tempfile.gettempdir()

    for filename in files.values():
        path = os.path.join(temp_dir, filename)

        if not os.path.exists(path):
            urllib.request.urlretrieve(
                base_url + filename,
                path
            )
    def load_images(path):
        with gzip.open(path, "rb") as f:
            images = np.frombuffer(
                f.read(),
                dtype=np.uint8,
                offset=16
            ).reshape(-1, 28, 28)

        return torch.tensor(
            images,
            dtype=torch.float32
        ) / 255.0

    def load_labels(path):
        with gzip.open(path, "rb") as f:
            labels = np.frombuffer(
                f.read(),
                dtype=np.uint8,
                offset=8
            )

        return torch.tensor(
            labels,
            dtype=torch.int64
        )

    X_train = load_images(
        os.path.join(temp_dir, files["train_images"])
    )[:n_train]

    y_train = load_labels(
        os.path.join(temp_dir, files["train_labels"])
    )[:n_train]

    X_test = load_images(
        os.path.join(temp_dir, files["test_images"])
    )[:n_test]

    y_test = load_labels(
        os.path.join(temp_dir, files["test_labels"])
    )[:n_test]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }

# Step 2 - FashionDataset (not yet solved)
# TODO: implement

# Step 3 - make_loaders (not yet solved)
# TODO: implement

# Step 4 - MLP (not yet solved)
# TODO: implement

# Step 5 - train_one_epoch (not yet solved)
# TODO: implement

# Step 6 - evaluate (not yet solved)
# TODO: implement

# Step 7 - fit (not yet solved)
# TODO: implement

# Step 8 - lr_range_test (not yet solved)
# TODO: implement

# Step 9 - random_search (not yet solved)
# TODO: implement

# Step 10 - test_accuracy (not yet solved)
# TODO: implement

# Step 11 - save_model (not yet solved)
# TODO: implement

# Step 12 - predict_classes (not yet solved)
# TODO: implement

