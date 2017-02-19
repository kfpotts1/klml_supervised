import numpy as np


def analyze(method, truths, predictions):
    """
    Analyzes the accuracy of a prediction against the known truths    
    
    syntax:
        output = analyze(method, truths, predictions)         

    Args:
        method (str): method for calculating accuracy
            'acc': classification accuracy
            'abs': absolute loss
            '01': 0/1 loss
            'squared': squared loss
        truths (numpy.array): 1xn array of correct classifications
        predictions (numpy.array): 1xn array of predicted classifications

    Returns:
        accuracy measure (float, int): measurement of accuracy or loss of predictions according to method
    """
    # this function only operates under the assumption that truths and predictions are numpy arrays

    truths = truths.flatten()
    predictions = predictions.flatten()

    if len(truths) != len(predictions):
        raise RuntimeError('len(truths) != len(predictions)')

    if method == 'abs':
        # compute the absolute difference between truths and predictions
        a = float(np.abs(truths - predictions).sum())
    elif method == 'acc':
        # a is the number of accurate predictions
        a = float(np.equal(truths, predictions).sum())
    elif method == '01':
        # a is the number of wrong predictions
        a = float((~np.equal(truths, predictions)).sum())
    elif method == 'squared':

        a = float(np.power(truths - predictions, 2).sum())
    else:
        raise ValueError('choose either "acc", "abs", "01", or "squared" for method')
    return a / float(len(truths))
