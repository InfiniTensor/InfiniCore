import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def addcdiv(
    input: Tensor,
    tensor1: Tensor,
    tensor2: Tensor,
    *,
    value=1.0,
    out=None,
) -> Tensor:
    r"""Apply the addcdiv function."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.addcdiv(
            input, tensor1, tensor2, value=value, out=out
        )

    if out is None:
        return Tensor(
            _infinicore.addcdiv(
                input._underlying,
                tensor1._underlying,
                tensor2._underlying,
                float(value),
            )
        )

    _infinicore.addcdiv_(
        input._underlying,
        tensor1._underlying,
        tensor2._underlying,
        out._underlying,
        float(value),
    )
    return out
