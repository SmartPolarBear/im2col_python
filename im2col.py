import numpy as np


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
    return kernel.reshape(c, h * w)


def conv2d(x: np.ndarray, kernels: np.ndarray, bias):
    c, h, w = kernels.shape
    _, xh, xw = x.shape

    img_flt = im2col(x, h, w)
    knl_flt = kernel_flattened(kernels)

    img_flt = img_flt.transpose((0, 2, 1))

    y = np.matmul(knl_flt, img_flt)
    y = np.sum(y, axis=0)
    y = y.reshape((c, xh - h + 1, xw - w + 1))
    return y + bias

# data = np.array(np.random.permutation(np.arange(stop=4 * 4 * 3)).reshape((3, 4, 4)), dtype=float)
# kernels = np.array(
#     np.random.permutation(np.arange(stop=3 * 3 * 4)).reshape((4, 3, 3)), dtype=float
# )
#
# # print(data, kernels)
#
# print(conv2d(data, kernels, 1.5))
