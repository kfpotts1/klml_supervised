#  K-Nearest Neighbors

from klml_supervised.distance import *
import numpy as np
from klml_supervised.train_test_split import uniform_split
from klml_supervised.analyze import *
from klml_supervised.stats import mode


def get_knn(x_train, x_test, k, dist_type='l2'):
    """
    Finds the k nearest neighbors of x_test in x_train.

    Args:
        x_train (numpy.ndarray): nxd input matrix with n row-vectors of dimensionality d
        x_test (numpy.ndarray): mxd input matrix with m row-vectors of dimensionality d
        k (int): number of nearest neighbors to be found
        dist_type (str, optional): type of distance metric, defaults to 'l2' for euclidean distance


    Returns:
        indices (numpy.ndarray): kxm matrix, where indices(i,j) is the i^th nearest neighbor of x_test(j,:)
        dists (numpy.ndarray): distances to the respective nearest neighbors
    """
    if dist_type in ['l2', 'L2', 'euclidean', 'Euclidean']:
        dists = l2distance(x_train, x_test).T  # Transpose here to loop over the **Test** point distances rather than tr
    else:
        raise ValueError('The distance measure selected is either not implemented or invalid')
    min_matrix = np.zeros((k, x_test.shape[0]))  # init as 0s
    index_matrix = np.zeros((k, x_test.shape[0]), dtype=int)  # needs to be int for slicing below! defaults to float

    if k > 0:
        for i in range(len(dists)):
            index_matrix[:, i] = dists[i].argsort()[:k]  # find indices of smallest distances
            min_matrix[:, i] = dists[i, (index_matrix[:, i])]  # find corresponding dists

    return index_matrix, min_matrix


def knn_classify(x_train, y_train, x_test, k='best', method='mode'):
    """
    k-nn classifier

    Determines the most likely labels for the given test data.

    Syntax:
        predictions = knn_classify(x_train,y_train,x_test,k)

    Args:
        x_train (numpy.ndarray): nxd input matrix with n row-vectors of dimensionality d
        y_train (numpy.array): nx1 labels for x_train points
        x_test (numpy.ndarray): mxd input matrix with m row-vectors of dimensionality d
        k (int, str):
            int: number of nearest neighbors to be found
            str: 'best' by default, automatically selects k

        method (str): method for selecting label
            'mode':  assigns an equal weighted vote for each nearest neighbor (most frequent neighbor, mode)
            'weighted': assigns a weighted vote for each nearest neighbor, larger votes for nearness
                - weighting scheme loosely based off
                [this site](http://www.data-machine.com/nmtutorial/distanceweightedknnalgorithm.htm)


    Returns:
        predictions (numpy.array): predicted labels, ie predictions(i) is the predicted label of x_test(i,:)
    """

    if k == 'best':
        # Split Test data into train and test
        X_train, X_test, Y_train, y_test = uniform_split(x_train, y_train)

        # we can argmax on k evaluated by our loss percentage
        best_k = (0, 0)
        for k_curr in range(1, 50):
            predictions = knn_classify(X_train, Y_train, X_test, k=k_curr, method=method)
            result = analyze("acc", y_test, predictions)
            if result > best_k[1]:
                best_k = (k_curr, result)
        predictions = knn_classify(x_train, y_train, x_test, k=best_k[0], method=method)
    else:
        indices_matrix, dist_matrix = get_knn(x_train, x_test, k)

        if method == 'weighted':
            predictions = np.zeros(indices_matrix.shape[1])
            for i in range(len(indices_matrix.T)):
                class_un = np.unique(y_train[(indices_matrix.T[i])].T)
                labels = y_train[(indices_matrix.T[i])].T
                scores = np.zeros(class_un.shape[0])
                for j in range(class_un.shape[0]):
                    idxs = np.argwhere(labels == class_un[j])[:, 1]
                    for idx in idxs:
                        scores[j] += (1 / dist_matrix.T[i, idx]**2)
                predictions[i] = class_un[np.argmax(scores)]
        elif method == 'mode':
            labels = np.zeros(indices_matrix.shape)
            for i in range(len(indices_matrix.T)):
                labels[:, i] = y_train[(indices_matrix.T[i])].T
            predictions = mode(labels)[0]  # simple mode calc,
            # if tie, takes least (maybe consider different k val in this case)
        else:
            raise ValueError('Invalid input for method.')

    return predictions

