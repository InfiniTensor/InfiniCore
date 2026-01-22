import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def var(
    input: Tensor,
    dim: int | tuple[int] | list[int] | None = None,
    unbiased: bool | None = None,
    correction: int | None = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out=None,
) -> Tensor:
    r"""Returns the variance of the input tensor."""

    if unbiased is not None:
        if correction is not None and correction != (1 if unbiased else 0):
            raise ValueError(
                "Cannot specify both 'unbiased' and 'correction' with conflicting values."
            )
        final_correction = 1 if unbiased else 0
    else:
        final_correction = correction if correction is not None else 1

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.var(
            input,
            dim=dim,
            correction=final_correction,
            keepdim=keepdim,
            dtype=dtype,
            out=out,
        )

    if dim is None:
        if out is None:
            return Tensor(_infinicore.var_global(input._underlying, final_correction))
        _infinicore.var_global_(input._underlying, out._underlying, final_correction)
        return out
    else:
        target_dim = dim
        if isinstance(target_dim, (tuple, list)):
            if len(target_dim) == 1:
                target_dim = target_dim[0]

        if out is None:
            return Tensor(
                _infinicore.var_reduce(
                    input._underlying, target_dim, final_correction, keepdim
                )
            )

        _infinicore.var_reduce_(
            input._underlying, out._underlying, target_dim, final_correction, keepdim
        )
        return out
