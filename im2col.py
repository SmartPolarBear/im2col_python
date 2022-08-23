import numpy as np

from numpy.lib.stride_tricks import as_strided


def im2col(x: np.ndarray, kernel_h, kernel_w) -> np.ndarray:
    """
    :param x: CHW shape
    :param kernel_w:
    :param kernel_h:
    :return:
    """
    xc, xh, xw = x.shape
    oh, ow = (xh - kernel_h + 1, xw - kernel_w + 1)

    elems = kernel_h * kernel_w

    wins = oh * ow

    y = np.zeros((xc, wins, elems))

    for row in range(xh - kernel_h + 1):
        for col in range(xw - kernel_w + 1):
            win = x[:, row:row + kernel_h, col:col + kernel_w]
            y[:, row * ow + col, :] = win.reshape((xc, elems))

    return y


def kernel_flattened(kernel: np.ndarray) -> np.ndarray:
    c, h, w = kernel.shape
    return kernel.reshape((c, 1, h * w))


def conv2d(x: np.ndarray, kernels: np.ndarray, bias):
    c, xh, xw = x.shape
    _, h, w = kernels.shape

    img_flt = im2col(x, h, w)
    knl_flt = kernel_flattened(kernels)

    y = np.einsum('cik,cjk->ij', knl_flt, img_flt)

    y = y.reshape((xh - h + 1, xw - w + 1))
    return y + bias


if __name__ == '__main__':
    data = np.array(np.arange(stop=2 * 4 * 4).reshape((2, 4, 4)), dtype=float)
    kernels = np.array(
        np.arange(stop=2 * 2 * 2).reshape((2, 2, 2)), dtype=float)

    # print(data, kernels)

    print(conv2d(data, kernels, 1.5))
