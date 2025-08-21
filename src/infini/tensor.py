def empty(*size, dtype=None, device=None):
    tensor = Tensor()

    tensor.shape = size

    tensor.strides = tuple(_calculate_default_strides(tensor.shape))

    tensor.dtype = dtype

    tensor.device = device

    return tensor


def from_torch(tensor):
    tensor_ = Tensor()

    tensor_.shape = tuple(tensor.shape)

    tensor_.strides = tuple(tensor.stride())

    return tensor_


class Tensor:
    empty = staticmethod(empty)

    from_torch = staticmethod(from_torch)


def _calculate_default_strides(shape):
    strides = [1]

    for size in reversed(shape[1:]):
        strides.append(size * strides[-1])

    return reversed(strides)
