
import pickle
import numpy as np
from typing import Type, Dict
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    KFold,
)
def load_mnist_dataset(
    nb_samples=None,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """
    Load the MNIST dataset.

    nb_samples: number of samples to save. Useful for code testing.
    The homework requires you to use the full dataset.

    Returns:
        X, y
        #X_train, y_train, X_test, y_test
    """

    try:
        # Are the datasets already loaded?
        print("... Is MNIST dataset local?")
        X: NDArray[np.floating] = np.load("mnist_X.npy")
        y: NDArray[np.int32] = np.load("mnist_y.npy", allow_pickle=True)
    except Exception as e:
        # Download the datasets
        print(f"load_mnist_dataset, exception {e}, Download file")
        X, y = datasets.fetch_openml(
            "mnist_784", version=1, return_X_y=True, as_frame=False
        )
        X = X.astype(float)
        y = y.astype(int)

    y = y.astype(np.int32)
    X: NDArray[np.floating] = X
    y: NDArray[np.int32] = y

    if nb_samples is not None and nb_samples < X.shape[0]:
        X = X[0:nb_samples, :]
        y = y[0:nb_samples]

    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)
    np.save("mnist_X.npy", X)
    np.save("mnist_y.npy", y)
    print('success')
    return X, y
load_mnist_dataset()