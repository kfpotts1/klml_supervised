import numpy as np


def prob_y(y):
    """
    Computes P(Y) given a label vector Y

    Uses plus one smoothing

    Syntax:
        positive_prob, negative_prob = prob_y(y);

    Args:
        y: (array_like)
            nx1 label vector, n labels (-1 or +1)

    Returns:
        pos: (float)
            probability p(y=1)
        neg: (float)
            probability p(y=-1)
    """

    # Plus one smoothing, MLE, avoid division error
    y = np.concatenate([y, [-1, 1]])
    n = y.shape[0]
    pos = np.exp(np.log(np.abs(np.sum(y[y == 1]))) - np.log(n))
    neg = np.exp(np.log(np.abs(np.sum(y[y == (-1)]))) - np.log(n))
    return pos, neg


def prob_x_given_y(x, y):
    """
    Computes P(X|Y) given training data x and labels y.

    Uses plus 1 smoothing. Does not compute true P(X|Y),
    only what is necessary for Naive Bayes functionality.

    Syntax:
        positive_prob, negative_prob = prob_x_given_y(x, y)

    Args:
        x: (array_like)
            n data vectors of d dimensions (nxd)
        y: (array_like)
            nx1 label vector, labels (-1 or +1) (nx1

    Returns:
        pos_prob:
            probability vector of p(x|y=1) (dx1)
        neg_prob:
            probability vector of p(x|y=-1) (dx1)
    """

    n, d = x.shape
    x = np.concatenate([x, np.ones((2, d))])
    y = np.concatenate([y, [-1, 1]])
    n, d = x.shape

    # needs to be replaced eventually (getting indices), numpy optimized
    pos_idx = []
    neg_idx = []
    [pos_idx.append(i) if y[i] == 1 else neg_idx.append(i) for i in range(n)]

    pos_idx = np.array(pos_idx)
    neg_idx = np.array(neg_idx)

    # Needs numpy optimization
    pos_x = x[(pos_idx), :]
    neg_x = x[(neg_idx), :]

    pos_prob = pos_x.sum(axis=0) / pos_x.sum()
    neg_prob = neg_x.sum(axis=0) / neg_x.sum()

    return pos_prob, neg_prob


def log_ratio_nb(x, y, x_test):
    """
    Uses Bayes rule to calculate the log ratio ln(P(Y = 1|X = x1)/P(Y = -1|X = x1))

    Syntax:
        ratio = log_ratio_nb(x,y, x_test);

    Args:
        x: (array_like)
            n input vectors of d dimensions (nxd)
        y: (array_like)
            n labels (-1 or +1)
        x_test: (array_like)
            test input vector(s) of d dimensions (1xd or nxd)

    Returns:
        log_ratio: (float or array_like)
            ln(P(Y = 1|X = x1)/P(Y = -1|X = x1))
    """

    # get proper numpy formatting
    x = np.asanyarray(x)
    y = np.asanyarray(y)
    x_test = np.asanyarray(x_test)

    # Calculate probabilities
    pos_prob, neg_prob = prob_x_given_y(x, y)
    p_pos, p_neg = prob_y(x, y)

    return np.dot(x_test, np.log(pos_prob)) - np.dot(x_test, np.log(neg_prob)) + np.log(p_pos) - np.log(p_neg)


def get_hyperplane(x, y):
    """
    Calculates the weight vector and bias for the hyperplane determined
    by the Naive Bayes method.

    Syntax:
        w,b= get_hyperplane(x,y)

    Args:
        x: (array_like)
            n input vectors of d dimensions (nxd)
        y: (array_like)
            label vector - n labels (-1 or +1)

    Returns:
        w: (array_like)
            weight vector of d dimensions
        b: (float)
            bias or hyperplane (scalar)
    """

    x = np.asanyarray(x)
    y = np.asanyarray(y)

    pos_prob, neg_prob = prob_x_given_y(x, y)
    p_pos, p_neg = prob_y(x, y)
    w = np.log(pos_prob) - np.log(neg_prob)
    b = np.log(p_pos) - np.log(p_neg)

    return w, b


def linear_classification(x, w, b=0):
    """
    classifies given x data points with the given hyperplane parameters

    Syntax:
        predictions = linear_classification(x, w, b);

    Args:
        x: (array_like)
            n input vectors of d dimensions (nxd)
        w: (array_like)
            weight vector of d dimensions
        b: (float)
            bias (optional)

    Returns:
        predictions: (array_like)
            predictions (-1 or +1)
    """

    predictions = np.dot(x, w)
    return np.asarray([1 if prediction + b > 0 else -1 for prediction in predictions])

