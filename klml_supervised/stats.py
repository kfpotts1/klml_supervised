# A collection of useful statistical functions for machine learning algorithms

import numpy as np


def mode(a, axis=0):
    """
    Returns the mode/s (most frequent element/s) of a

    Note:
        - This mode function uses a quicksort, which may be inefficient for large arrays
        - Returns None if len(a) <= 0

    Args:
        a: array_like
            will find the mode (most frequent element) of a
        axis: int
            if nxm array, axis for mode calculation
            defaults to 0

    Returns:
        m: float or array_like
            mode (most frequent element) of a
        frequency: float or array_like
            the frequency (number of occurrences) of the mode
    """
    a = np.asanyarray(a.copy())
    if a.shape[0] == a.size:  # is 1xn or nx1
        if a.size == 0:
            return None
        else:
            max_freq = 1
            a.sort()
            m = a[0]
            curr_freq = 1
            for i in range(1, len(a)):
                if a[i] == a[i - 1]:
                    curr_freq += 1
                if curr_freq > max_freq:
                    max_freq = curr_freq
                    m = a[i - 1]
                if a[i] != a[i - 1]:
                    curr_freq = 1
            return m, max_freq
    else:  # nxm array
        if axis == 0:
            a = a.T
        l = len(a)
        m = np.zeros(l)
        f = np.zeros(l)
        for i in range(l):
            m[i], f[i] = mode(a[i])
        return m, f
