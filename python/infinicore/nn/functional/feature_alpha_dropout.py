import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def feature_alpha_dropout(
    input: Tensor,
    p: float = 0.5,
    training: bool = False,
    inplace: bool = False,
    *,
    out=None,
) -> Tensor:
    r"""Apply feature alpha dropout.

    Equivalent to torch.nn.functional.feature_alpha_dropout.
    """

    if p < 0.0 or p >= 1.0:
        raise ValueError(
            f"dropout probability has to satisfy 0 <= p < 1, but got {p}"
        )

    if input.ndim < 2:
        raise RuntimeError("Feature dropout requires at least 2 dimensions in the input")

    if inplace and out is not None:
        raise RuntimeError("`inplace=True` and `out` cannot be used together.")

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.feature_alpha_dropout(
            input,
            p=p,
            training=training,
            inplace=inplace,
        )

    if inplace:
        _infinicore.feature_alpha_dropout_(
            input._underlying,
            input._underlying,
            float(p),
            bool(training),
        )
        return input

    if out is None:
        return Tensor(
            _infinicore.feature_alpha_dropout(
                input._underlying,
                float(p),
                bool(training),
            )
        )

    _infinicore.feature_alpha_dropout_(
        out._underlying,
        input._underlying,
        float(p),
        bool(training),
    )

    return out