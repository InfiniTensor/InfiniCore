import infinicore.device
import infinicore.dtype
from infinicore.lib import _infinicore


class Tensor:
    def __init__(self, underlying):
        """An internal method. Please do not use this directly."""

        self._underlying = underlying

        self._dtype = infinicore.dtype(self._underlying.dtype)

        self._device = infinicore.device._from_infinicore_device(
            self._underlying.device
        )

    @property
    def shape(self):
        return self._underlying.shape

    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self):
        return self._device

    @property
    def ndim(self):
        return self._underlying.ndim

    def data_ptr(self):
        return self._underlying.data_ptr()

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
        self._underlying.copy_(src._underlying)

    def to(self, *args, **kwargs):
        return Tensor(
            self._underlying.to(*tuple(arg._underlying for arg in args), **kwargs)
        )

    def as_strided(self, size, stride):
        return Tensor(self._underlying.as_strided(size, stride))

    def contiguous(self):
        return Tensor(self._underlying.contiguous())

    def permute(self, dims):
        return Tensor(self._underlying.permute(dims))

    def view(self, shape):
        return Tensor(self._underlying.view(shape))

    def debug(self, filename=None):
        """Print tensor data or save to file for debugging
        
        Args:
            filename: Optional filename to save raw binary data. If None, prints to stdout.
        """
        if filename is None:
            self._underlying.debug()
        else:
            self._underlying.debug(filename)


def empty(size, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.empty(size, dtype._underlying, device._underlying, pin_memory)
    )


def strided_empty(size, strides, *, dtype=None, device=None, pin_memory=False):
    return Tensor(
        _infinicore.strided_empty(
            size, strides, dtype._underlying, device._underlying, pin_memory
        )
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


def strided_from_blob(data_ptr, size, strides, *, dtype=None, device=None):
    return Tensor(
        _infinicore.strided_from_blob(
            data_ptr, size, strides, dtype._underlying, device._underlying
        )
    )
