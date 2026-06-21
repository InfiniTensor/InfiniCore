import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def scatter_add(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
    *,
    out=None,
) -> Tensor:
    r"""Add all values from src into input at the indices specified in index along dim."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.scatter_add(
            input,
            dim,
            index,
            src,
            out=out,
        )

    if out is None:
        if not hasattr(_infinicore, "scatter_add"):
            raise NotImplementedError(
                "scatter_add is not implemented in _infinicore, "
                "and ntops path is unavailable."
            )

        return Tensor(
            _infinicore.scatter_add(
                input._underlying,
                dim,
                index._underlying,
                src._underlying,
            )
        )

    if not hasattr(_infinicore, "scatter_add_"):
        raise NotImplementedError(
            "scatter_add_ out variant is not implemented in _infinicore."
        )

    _infinicore.scatter_add_(
        out._underlying,
        input._underlying,
        dim,
        index._underlying,
        src._underlying,
    )

    return out


def scatter_add_(
    input: Tensor,
    dim: int,
    index: Tensor,
    src: Tensor,
) -> Tensor:
    r"""In-place scatter_add."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.scatter_add(
            input,
            dim,
            index,
            src,
            out=input,
        )

    if not hasattr(_infinicore, "scatter_add_"):
        raise NotImplementedError(
            "scatter_add_ is not implemented in _infinicore, "
            "and ntops path is unavailable."
        )

    _infinicore.scatter_add_(
        input._underlying,
        input._underlying,
        dim,
        index._underlying,
        src._underlying,
    )

    return input