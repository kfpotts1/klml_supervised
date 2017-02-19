import numpy as np


def innerproduct(x, z=None):
    """
    Computes the inner-product matrix.

    syntax:
        d = innerproduct(x,z)
    Args:
        x (numpy.ndarray): nxd data matrix with n vectors (rows) of dimensionality d
        z (numpy.ndarray): mxd data matrix with m vectors (rows) of dimensionality d

    Output:
        g (numpy.ndarray): Matrix g of size nxm
            g[i,j] is the inner-product between vectors x[i,:] and z[j,:]

    When called with only one input:
        innerproduct(x) = innerproduct(x,x)
    """
    if z is None:  # case when there is only one input (x)
        g = np.dot(x, x.transpose())
    else:  # case when there are two inputs (x,z)
        z = z.transpose()
        if x.shape[1] != z.shape[0]:
            raise ValueError("Invalid Matrices")
        g = np.dot(x, z)
    return g


def l2distance(x, z=None):
    """
    function d = l2distance(x,z)

    Computes the Euclidean distance matrix.
    syntax:
        d=l2distance(x,z)
    Args:
        x (numpy.dnarray): nxd data matrix with n vectors (rows) of dimensionality d
        z (numpy.ndarray): mxd data matrix with m vectors (rows) of dimensionality d

    Returns:
        d (numpy.ndarray): Matrix d of size nxm
            d(i,j) is the Euclidean distance of x(i,:) and z(j,:)

    When called with only one input:
        l2distance(x) = l2distance(x,x)
    """

    if z is None:
        return l2distance(x, x)
    else:  # case when there are two inputs (x,z)
        s = (x*x).sum(axis=1).reshape(-1, 1)
        r = (z*z).sum(axis=1)
        g = innerproduct(x, z)
        d_2 = s - 2*g + r
        if x is z:
            np.fill_diagonal(d_2, 0.0)
        d = np.sqrt(d_2)
        return d
