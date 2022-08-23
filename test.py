import numpy as np
import numbers

from numpy.lib.stride_tricks import as_strided


def view_as_windows(arr_in: np.ndarray, window_shape, step=1):
    ndim = arr_in.ndim

    if isinstance(window_shape, numbers.Number):
        window_shape = (window_shape,) * ndim

    if not (len(window_shape) == ndim):
        raise ValueError("`window_shape` is incompatible with `arr_in.shape`")

    if isinstance(step, numbers.Number):
        if step < 1:
            raise ValueError("`step` must be >= 1")
        step = (step,) * ndim
    if len(step) != ndim:
        raise ValueError("`step` is incompatible with `arr_in.shape`")

    arr_shape = np.array(arr_in.shape)
    window_shape = np.array(window_shape, dtype=arr_shape.dtype)

    if ((arr_shape - window_shape) < 0).any():
        raise ValueError("`window_shape` is too large")

    if ((window_shape - 1) < 0).any():
        raise ValueError("`window_shape` is too small")

    # -- build rolling window view
    slices = tuple(slice(None, None, st) for st in step)
    window_strides = np.array(arr_in.strides)

    indexing_strides = arr_in[slices].strides

    win_indices_shape = (((np.array(arr_in.shape) - np.array(window_shape))
                          // np.array(step)) + 1)

    new_shape = tuple(list(win_indices_shape) + list(window_shape))
    strides = tuple(list(indexing_strides) + list(window_strides))
    arr_out = as_strided(arr_in, shape=new_shape, strides=strides)
    return arr_out


data = np.array(np.random.permutation(np.arange(stop=4 * 4 * 3)).reshape((3, 4, 4)), dtype=np.int64)
# print(data)
print(view_as_windows(data, (3, 3, 3)).shape)
