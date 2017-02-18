#  K-Nearest Neighbors

from klml_supervised.distance import *


def get_knn(x_train, x_test, k, dist_type='l2'):
    """
    function [indices,dists] = get_knn(x_train,x_test,k);

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

