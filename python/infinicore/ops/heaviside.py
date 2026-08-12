import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def heaviside(input: Tensor, values: Tensor, *, out=None) -> Tensor:
    r"""Compute the Heaviside step function for each element in input."""
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.heaviside(input, values, out=out)
    if out is None:
        if hasattr(_infinicore, "heaviside"):
            return Tensor(_infinicore.heaviside(input._underlying, values._underlying))

        raise NotImplementedError(
            "heaviside is only implemented through ntops on cuda/musa; "
            "_infinicore.heaviside is not available."
        )

    if hasattr(_infinicore, "heaviside_"):
        _infinicore.heaviside_(out._underlying, input._underlying, values._underlying)
        return out

    raise NotImplementedError(
        "heaviside out= path is only implemented through ntops on cuda/musa; "
        "_infinicore.heaviside_ is not available."
    )