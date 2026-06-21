import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _copy_result(out: Tensor, res: Tensor) -> Tensor:
    copy_fn = getattr(out._underlying, "copy_", None)

    if callable(copy_fn):
        copy_fn(res._underlying)
        return out

    raise NotImplementedError(
        "frac requires underlying tensor copy_ for inplace/out fallback."
    )


def frac(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    r"""Compute the fractional portion of each element in input."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        res = infinicore.ntops.torch.frac(input)

        if inplace:
            return _copy_result(input, res)

        if out is not None:
            return _copy_result(out, res)

        return res

    if inplace:
        if hasattr(_infinicore, "frac_"):
            _infinicore.frac_(input._underlying, input._underlying)
            return input

        if hasattr(_infinicore, "frac"):
            res = Tensor(_infinicore.frac(input._underlying))
            return _copy_result(input, res)

        raise NotImplementedError(
            "frac inplace requires ntops backend, `_infinicore.frac_`, "
            "or `_infinicore.frac` with copy_ support."
        )

    if out is None:
        if hasattr(_infinicore, "frac"):
            return Tensor(_infinicore.frac(input._underlying))

        raise NotImplementedError(
            "frac requires ntops backend or `_infinicore.frac`."
        )

    if hasattr(_infinicore, "frac_"):
        _infinicore.frac_(out._underlying, input._underlying)
        return out

    if hasattr(_infinicore, "frac"):
        res = Tensor(_infinicore.frac(input._underlying))
        return _copy_result(out, res)

    raise NotImplementedError(
        "frac out requires ntops backend, `_infinicore.frac_`, "
        "or `_infinicore.frac` with copy_ support."
    )