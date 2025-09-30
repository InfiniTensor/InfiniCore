import ctypes

import _infinicore


class Tensor:
    def __init__(self, tensor):
        """An internal method. Please do not use this directly."""

        self._underlying = tensor

    @property
    def shape(self):
        return self._underlying.shape

    @property
    def dtype(self):
        return self._underlying.dtype

    @property
    def device(self):
        return self._underlying.device

    @property
    def ndim(self):
        return self._underlying.device

    def data_ptr(self):
        return self._underlying.data_ptr

    def size(self, dim=None):
        if dim is None:
            return self.shape

        return self.shape[dim]

    def stride(self, dim=None):
        if dim is None:
            return self._underlying.strides

        return self._underlying.strides[dim]

    def numel(self):
        return self._underlying.numel()

    def is_contiguous(self):
        return self._underlying.is_contiguous()

    def is_is_pinned(self):
        return self._underlying.is_is_pinned()

    def copy_(self, src):
        return Tensor(self._underlying.copy_(src._underlying))

    def to(self, *args, **kwargs):
        return Tensor(
            self._underlying.to(*tuple(arg._underlying for arg in args), **kwargs)
        )

    def as_strided(self, size, stride):
        Tensor(self._underlying.as_strided(size, stride))

    def contiguous(self):
        return Tensor(self._underlying.contiguous())

    def permute(self, dims):
        return Tensor(self._underlying.permute(dims))

    def view(self, shape):
        return Tensor(self._underlying.view(shape))


def empty(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.empty(size, dtype._underlying, device._underlying, pin_memory)
    )


def zeros(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.zeros(size, dtype._underlying, device._underlying, pin_memory)
    )


def ones(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.ones(size, dtype._underlying, device._underlying, pin_memory)
    )


def from_blob(data_ptr, size, *, dtype=None, device=None):
    return Tensor(
        _infinicore.from_blob(data_ptr, size, dtype._underlying, device._underlying)
    )
