import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def var_mean(
    input: Tensor,
    dim: int | tuple[int] | list[int] | None = None,
    unbiased: bool | None = None,
    correction: int | None = None,
    keepdim: bool = False,
    *,
    dtype=None,
    out=None,
) -> tuple[Tensor, Tensor]:
    r"""Calculates the variance and mean of input tensor."""

    if unbiased is not None:
        if correction is not None and correction != (1 if unbiased else 0):
            raise ValueError(
                "Cannot specify both 'unbiased' and 'correction' with conflicting values."
            )
        final_correction = 1 if unbiased else 0
    else:
        final_correction = correction if correction is not None else 1

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.var_mean(
            input,
            dim=dim,
            correction=final_correction,
            keepdim=keepdim,
            dtype=dtype,
            out=out,
        )

    if dim is None:
        if out is None:
            v_tensor, m_tensor = _infinicore.var_mean_global(
                input._underlying, final_correction
            )
            return Tensor(v_tensor), Tensor(m_tensor)

        if not isinstance(out, (list, tuple)) or len(out) < 2:
            raise ValueError("out must be a tuple/list of two Tensors for var_mean")

        _infinicore.var_mean_global_(
            input._underlying, out[0]._underlying, out[1]._underlying, final_correction
        )
        return out[0], out[1]

    else:
        target_dim = dim
        if isinstance(target_dim, (tuple, list)):
            if len(target_dim) == 1:
                target_dim = target_dim[0]

        if out is None:
            v_tensor, m_tensor = _infinicore.var_mean_reduce(
                input._underlying, target_dim, final_correction, keepdim
            )
            return Tensor(v_tensor), Tensor(m_tensor)

        if not isinstance(out, (list, tuple)) or len(out) < 2:
            raise ValueError("out must be a tuple/list of two Tensors for var_mean")

        _infinicore.var_mean_reduce_(
            input._underlying,
            out[0]._underlying,
            out[1]._underlying,
            target_dim,
            final_correction,
            keepdim,
        )
        return out[0], out[1]
