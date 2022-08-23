import numpy as np


def conv2d(x, kernels, bias):
    """
    :param x: CHW shape
    :param kernels: NCHW shape
    :param bias: scalar
    :return:
    """
    c, xh, xw = x.shape
    _, kh, kw = kernels.shape

    oh, ow = (xh - kh + 1, xw - kw + 1)

    y = np.zeros((oh, ow))
    for row in range(xh - kh + 1):
        for col in range(xw - kw + 1):
            win = x[:, row:row + kh, col:col + kw]
            y[row, col] = np.sum(win * kernels)

    return y + bias
