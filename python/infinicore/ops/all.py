import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def all(
    input: Tensor,
    dim: int | tuple[int] | list[int] | None = None,
    keepdim: bool = False,
    *,
    out=None,
) -> Tensor:
    r"""Computes the logical AND of all elements."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.all(input, dim=dim, keepdim=keepdim, out=out)

    if dim is None:
        if out is None:
            return Tensor(_infinicore.all_global(input._underlying))
        _infinicore.all_global_(input._underlying, out._underlying)
        return out

    else:
        if isinstance(dim, int):
            dims = [dim]
        else:
            dims = list(dim)

        ndim = input.ndim
        normalized_dims = sorted(
            [d if d >= 0 else d + ndim for d in dims], reverse=True
        )

        current_input = input

        if len(normalized_dims) == 1 and out is not None:
            _infinicore.all_reduce_(
                current_input._underlying, out._underlying, normalized_dims[0], keepdim
            )
            return out

        for i, target_dim in enumerate(normalized_dims):
            is_last_step = i == len(normalized_dims) - 1

            if is_last_step and out is not None:
                _infinicore.all_reduce_(
                    current_input._underlying, out._underlying, target_dim, keepdim
                )
                return out
            else:
                res_ptr = _infinicore.all_reduce(
                    current_input._underlying, target_dim, keepdim
                )
                current_input = Tensor(res_ptr)

        return current_input
