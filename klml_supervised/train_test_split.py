# functions for splitting data into training and testing data sets
import numpy as np


def uniform_split(x, y, train_size=0.66, prng_seed=42):
    """
    Splits data (x) and labels (y) up into training and testing data sets

    Uniformly shuffles data and labels. Does not consider y labels into separation scheme.

    Args:
        x: (array_like)
            nxd array of data
        y: (array_like)
            nx1 array of data labels
        train_size: (float), optional, defaults to 0.66
            determines the proportion used returned as training data
            (1 - train_size) used as test size
        prng_seed: (int)
            seed for numpy PRNG used for random shuffle
    """
    x = np.asanyarray(x)
    y = np.asanyarray(y).flatten()
    n = len(x)
    indices = np.arange(0, n)

    np.random.seed(seed=prng_seed)
    np.random.shuffle(indices)  # shuffles inplace

    x_shuff = x[(indices), :]
    y_shuff = y[(indices)]

    num_trains = int(n * train_size)

    x_train = x_shuff[:num_trains, :]
    x_test = x_shuff[num_trains:, :]
    y_train = y_shuff[:num_trains]
    y_test = y_shuff[num_trains:]

    return x_train, x_test, y_train, y_test
