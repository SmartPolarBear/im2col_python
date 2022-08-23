import numpy as np


def conv2d(x, kernels, bias):
    """
    :param x: CHW shape
    :param kernels: CHW shape
    :param bias: scalar
    :return:
    """
    xc, xh, xw = x.shape
    kc, kh, kw = kernels.shape

    oh, ow = (xh - kh + 1, xw - kw + 1)

    y = np.zeros((kc, oh, ow))
    for ftl in range(kc):
        kernel = kernels[ftl, :]
        for row in range(xh - kh + 1):
            for col in range(xw - kw + 1):
                win = x[:, row:row + kh, col:col + kw]
                y[ftl, row, col] = np.sum(win * kernel)

    return y + bias
