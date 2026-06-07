import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fractional_max_pool2d(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
) -> Tensor:
    r"""Apply 2D fractional max pooling over an input signal."""

    assert input.ndim == 4, (
        "`fractional_max_pool2d` only supports 4D input for now."
    )

    assert not return_indices, (
        "`return_indices` is not supported by ntops fractional_max_pool2d yet."
    )

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.fractional_max_pool2d(
            input,
            kernel_size=kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )

    if hasattr(_infinicore, "fractional_max_pool2d"):
        if _random_samples is None:
            return Tensor(
                _infinicore.fractional_max_pool2d(
                    input._underlying,
                    kernel_size,
                    output_size,
                    output_ratio,
                    return_indices,
                )
            )

        return Tensor(
            _infinicore.fractional_max_pool2d(
                input._underlying,
                kernel_size,
                output_size,
                output_ratio,
                return_indices,
                _random_samples._underlying,
            )
        )
