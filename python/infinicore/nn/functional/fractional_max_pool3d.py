import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fractional_max_pool3d(
    input: Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices: bool = False,
    _random_samples=None,
) -> Tensor:
    r"""Apply 3D fractional max pooling over an input signal."""

    assert input.ndim == 5, (
        "`fractional_max_pool3d` only supports 5D input for now."
    )

    assert not return_indices, (
        "`return_indices` is not supported by ntops fractional_max_pool3d yet."
    )

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.fractional_max_pool3d(
            input,
            kernel_size=kernel_size,
            output_size=output_size,
            output_ratio=output_ratio,
            return_indices=return_indices,
            _random_samples=_random_samples,
        )

    if hasattr(_infinicore, "fractional_max_pool3d"):
        if _random_samples is None:
            return Tensor(
                _infinicore.fractional_max_pool3d(
                    input._underlying,
                    kernel_size,
                    output_size,
                    output_ratio,
                    return_indices,
                )
            )

        return Tensor(
            _infinicore.fractional_max_pool3d(
                input._underlying,
                kernel_size,
                output_size,
                output_ratio,
                return_indices,
                _random_samples._underlying,
            )
        )