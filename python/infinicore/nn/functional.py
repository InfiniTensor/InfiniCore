import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

__all__ = ["causal_softmax", "random_sample", "rms_norm", "silu", "swiglu"]


def causal_softmax(input: Tensor, out=None) -> Tensor:
    r"""Apply a causal softmax function."""

    if out is None:
        return Tensor(_infinicore.causal_softmax(input._underlying))

    _infinicore.causal_softmax_(out._underlying, input._underlying)

    return out


def rms_norm(
    input: Tensor,
    normalized_shape: list[int],
    weight: Tensor,
    eps: float = 1e-5,
    *,
    out=None,
) -> Tensor:
    r"""Apply Root Mean Square Layer Normalization."""

    assert normalized_shape == weight.shape, (
        "normalized_shape does not match weight.shape."
    )

    if out is None:
        return Tensor(_infinicore.rms_norm(input._underlying, weight._underlying, eps))

    _infinicore.rms_norm_(out._underlying, input._underlying, weight._underlying, eps)

    return out


def silu(input: Tensor, inplace: bool = False, *, out=None) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.silu(input, inplace=inplace)

    if inplace:
        _infinicore.silu_(input._underlying, input._underlying)
        return input

    if out is None:
        return Tensor(_infinicore.silu(input._underlying))

    _infinicore.silu_(out._underlying, input._underlying)

    return out


def swiglu(input: Tensor, other: Tensor, *, out=None):
    r"""Apply the Swish-Gated Linear Unit (SwiGLU) function, element-wise."""

    if out is None:
        return Tensor(_infinicore.swiglu(input._underlying, other._underlying))

    _infinicore.swiglu_(out._underlying, input._underlying, other._underlying)

    return out


def linear(input: Tensor, weight: Tensor, bias=None, *, out=None) -> Tensor:
    r"""Applies a linear transformation to the incoming data: y=xA^T+b."""

    if out is None:
        return Tensor(
            _infinicore.linear(
                input._underlying,
                weight._underlying,
                None if bias is None else bias._underlying,
            )
        )

    _infinicore.linear_(
        out._underlying,
        input._underlying,
        weight._underlying,
        None if bias is None else bias._underlying,
    )
def random_sample(
    logits: Tensor,
    random_val: float,
    topp: float,
    topk: int,
    temperature: float,
    *,
    out=None,
) -> Tensor:
    r"""Sample an index from logits with nucleus/top-k filtering."""

    if out is None:
        return Tensor(
            _infinicore.random_sample(
                logits._underlying,
                random_val,
                topp,
                topk,
                temperature,
            )
        )

    _infinicore.random_sample_(
        out._underlying,
        logits._underlying,
        random_val,
        topp,
        topk,
        temperature,
    )

    return out
