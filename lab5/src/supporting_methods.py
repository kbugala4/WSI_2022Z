import numpy as np


def ReLu(x, is_deriv=False):
    if is_deriv:
        return 1 * (x >= 0)
    else:
        return x * (x >= 0)


def sigmoid(x, is_deriv=False):
    f = 1.0 / (1.0 + np.exp(-x))
    if is_deriv:
        return f * (f - 1)
    else:
        return f


def softmax(x, is_deriv=False):
    exp = np.exp(x - x.max())
    if is_deriv:
        return exp / np.sum(exp, axis=0) * (1 - exp / np.sum(exp, axis=0))
    return exp / np.sum(exp, axis=0)
