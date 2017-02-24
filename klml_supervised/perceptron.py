# implements a simple binary (2 label) perceptron classification algorithms
# assumes that the data is linearly separable

import numpy as np


def update_weight(x, y, w):
    """
    Updates the weight vector w based on x and y.

    Caution:
        does not check for np.dot(w, x) * y <= 0, instead this is assumed.
        In the simple perceptron alg, the update should only be performed
        when np.dot(w, x) * y <= 0.

    Syntax
    w = update_weight(x,y,w)

    Args:
        x: (array_like)
            input vector of d dimensions (1xd)
        y: (int, {-1, 1})
            corresponding label (-1 or +1)
        w: (array_like)
            weight vector before updating

    Returns:
    w: (array_like)
        weight vector after updating
    """
    # just in case x, w are accidentally transposed (prevents future bugs)
    x = np.asanyarray(x).flatten()
    w = np.asanyarray(w).flatten()

    if y not in {-1, 1} and y not in {-1.0, 1.0}:
        raise ValueError('Invalid label y, should be -1 or 1')

    return np.add(w, x * y)


def find_hyperplane(x, y):
    """
    Finds the normal vector w which defines the separating hyperplane of the data.

    Syntax:
        w = find_hyperplane(x, y)

    Args:
        x: (matrix_like)
            n input vectors with d dimensions each (nxd matrix)
    y: (array_like)
        label vector (nx1) {-1 or +1}

    Returns:
        w: (array_like)
            weight vector (1xd) for the separating hyperplane
    """
    MAX_ITERATION = 200

    n, d = x.shape
    w = np.zeros((1, d))
    index = list(range(0, len(x)))
    for j in range(MAX_ITERATION):
        m = 0
        np.random.shuffle(index)
        for i in index:
            if y[i] * np.dot(w, x[i]) <= 0:
                w = update_weight(x[i], y[i], w)
                m += 1
        if m == 0:
            return w
    raise RuntimeError('Data appears to be not linearly separable,' +
                       ' failed to converge after {0} iterations'.format(MAX_ITERATION))


def linear_classification(x, w, b=0):
    """
    Determines the predicted labels for x, given the weight vector w.


    Syntax:
        predictions = linear_classification(x, w, b)

    Args:
        x: (matrix_like)
            n input vectors with d dimensions (dxn matrix)
        w: (array_like)
            weight vector (dx1), defines a hyperplane which linearly separates test points
        b: (int, float)
            bias, also known as hyperplane intercept.

    Returns:
        preds: (array_like)
            predicted label (1xn)
    """
    # TODO: address bias b
    w = w.reshape(-1)
    dot = np.dot(x, w)

    dot = np.asanyarray(dot)
    dot[dot > 0] = 1
    dot[dot <= 0] = -1
    return dot
