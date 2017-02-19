# A collection of useful statistical functions for machine learning algorithms

import numpy as np


def mode(a):
    """
    Returns the mode (most frequent element) of a

    Note:
        - This mode function uses a quicksort, which may be inefficient for large arrays
        - Returns None if len(a) <= 0

    Args:
        a: array_like
            will find the mode (most frequent element) of a

    Returns:
        m: int or float
            mode (most frequent element) of a
        frequency: int
            the frequency (number of occurrences) of the mode
    """
    a = np.asanyarray(a.copy())
    if len(a) > 0:
        max_freq = 1
        a.sort()
        m = a[0]
        curr_freq = 1
        for i in range(1, len(a)):
            if a[i] == a[i-1]:
                curr_freq += 1
            else:
                if curr_freq > max_freq:
                    max_freq = curr_freq
                    m = a[i-1]
                curr_freq = 1
        return m, max_freq
    return None
