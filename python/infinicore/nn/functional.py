import infinicore
from infinicore.lib import _infinicore

__all__ = ["causal_softmax", "rms_norm", "silu", "swiglu"]


def causal_softmax(
        input: infinicore.Tensor,
        out=None
) -> infinicore.Tensor:
    r"""Apply a causal softmax function.
    """

    if out is None:
        return infinicore.Tensor(_infinicore.causal_softmax(input._underlying))

    _infinicore.causal_softmax_(out._underlying, input._underlying)

    return out


def rms_norm(
        input: infinicore.Tensor,
        normalized_shape: list[int],
        weight: infinicore.Tensor,
        eps: float = 1e-5,
        out=None
) -> infinicore.Tensor:
    r"""Apply Root Mean Square Layer Normalization.
    """

    assert normalized_shape == weight.shape, "normalized_shape does not match weight.shape."

    if out is None:
        return infinicore.Tensor(
            _infinicore.rms_norm(input._underlying, weight._underlying, eps)
        )

    _infinicore.rms_norm_(out._underlying, input._underlying, weight._underlying, eps)

    return out


def silu(input: infinicore.Tensor, inplace: bool = False, out=None) -> infinicore.Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise.
    """

    if infinicore.use_ntops and out is None:
        return infinicore.ntops.torch.silu(input, inplace=inplace)

    if inplace:
        _infinicore.silu_(input._underlying, input._underlying)
        return input

    if out is None:
        return infinicore.Tensor(_infinicore.silu(input._underlying))

    _infinicore.silu_(out._underlying, input._underlying)

    return out


def swiglu(input: infinicore.Tensor, other: infinicore.Tensor, out=None):
    r"""Apply the Swish-Gated Linear Unit (SwiGLU) function, element-wise.
    """

    if out is None:
        return infinicore.Tensor(_infinicore.swiglu(input._underlying, other._underlying))

    _infinicore.swiglu_(out._underlying, input._underlying, other._underlying)

    return out
