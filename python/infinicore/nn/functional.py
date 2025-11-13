from typing import Optional

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


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Optional[Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    *,
    out=None,
) -> Tensor:
    r"""Computes scaled dot product attention on query, key and value tensors."""
    assert (attn_mask is None) and (0.0 == dropout_p), "Unsupported parameters."
    assert (enable_gqa is True) and (is_causal is True), "Incorrect parameter value."

    ntoken = query.shape[-2]
    total_token = key.shape[-2]

    assert (1 == ntoken and total_token > 1) or (ntoken == total_token), (
        "Incorrect parameter value."
    )

    if out is None:
        return infinicore.Tensor(
            _infinicore.scaled_dot_product_attention(
                query._underlying, key._underlying, value._underlying, scale
            )
        )

    _infinicore.scaled_dot_product_attention_(
        out._underlying,
        query._underlying,
        key._underlying,
        value._underlying,
        scale,
def embedding(input: Tensor, weight: Tensor, *, out=None) -> Tensor:
    r"""Generate a simple lookup table that looks up embeddings in a fixed dictionary and size."""

    if out is None:
        return Tensor(_infinicore.embedding(input._underlying, weight._underlying))

    _infinicore.embedding_(out._underlying, input._underlying, weight._underlying)
    return out


def rope(
    x: Tensor,
    pos_ids: Tensor,
    sin_table: Tensor,
    cos_table: Tensor,
    algo: _infinicore.Algo = _infinicore.Algo.GPT_NEOX,
    *,
    out=None,
) -> Tensor:
    r"""Rotary Position Embedding(RoPE)."""

    if out is None:
        return infinicore.Tensor(
            _infinicore.rope(
                x._underlying,
                pos_ids._underlying,
                sin_table._underlying,
                cos_table._underlying,
                algo,
            )
        )

    _infinicore.rope_(
        out._underlying,
        x._underlying,
        pos_ids._underlying,
        sin_table._underlying,
        cos_table._underlying,
        algo,
    )
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
