import contextlib

with contextlib.suppress(ImportError):
    from ._preload import preload

    preload()

import infinicore.context as context
import infinicore.nn as nn

# Import context functions
from infinicore.context import (
    get_device,
    get_device_count,
    get_stream,
    is_graph_recording,
    set_device,
    start_graph_recording,
    stop_graph_recording,
    sync_device,
    sync_stream,
)
from infinicore.device import device
from infinicore.device_event import DeviceEvent
from infinicore.dtype import (
    bfloat16,
    bool,
    cdouble,
    cfloat,
    chalf,
    complex32,
    complex64,
    complex128,
    double,
    dtype,
    float,
    float16,
    float32,
    float64,
    half,
    int,
    int8,
    int16,
    int32,
    int64,
    long,
    short,
    uint8,
)
from infinicore.ops.add import add
from infinicore.ops.add_rms_norm import add_rms_norm
from infinicore.ops.addcmul import addcmul
from infinicore.ops.addr import addr
from infinicore.ops.all import all
from infinicore.ops.argwhere import argwhere
from infinicore.ops.asin import asin
from infinicore.ops.asinh import asinh
from infinicore.ops.atanh import atanh
from infinicore.ops.attention import attention
from infinicore.ops.baddbmm import baddbmm
from infinicore.ops.bilinear import bilinear
from infinicore.ops.binary_cross_entropy_with_logits import (
    binary_cross_entropy_with_logits,
)
from infinicore.ops.cat import cat
from infinicore.ops.cdist import cdist
from infinicore.ops.cosh import cosh
from infinicore.ops.cross_entropy import cross_entropy
from infinicore.ops.equal import equal
from infinicore.ops.fmin import fmin
from infinicore.ops.fmod import fmod
from infinicore.ops.hypot import hypot
from infinicore.ops.index_add import index_add
from infinicore.ops.index_copy import index_copy
from infinicore.ops.inner import inner
from infinicore.ops.kv_caching import kv_caching
from infinicore.ops.masked_select import masked_select
from infinicore.ops.matmul import matmul
from infinicore.ops.mha_kvcache import mha_kvcache
from infinicore.ops.mha_varlen import mha_varlen
from infinicore.ops.mul import mul
from infinicore.ops.narrow import narrow
from infinicore.ops.paged_attention import paged_attention
from infinicore.ops.paged_attention_prefill import paged_attention_prefill
from infinicore.ops.paged_caching import paged_caching
from infinicore.ops.rearrange import rearrange
from infinicore.ops.reciprocal import reciprocal
from infinicore.ops.round import round
from infinicore.ops.squeeze import squeeze
from infinicore.ops.sum import sum
from infinicore.ops.take import take
from infinicore.ops.tan import tan
from infinicore.ops.topk import topk
from infinicore.ops.unsqueeze import unsqueeze
from infinicore.ops.var import var
from infinicore.ops.var_mean import var_mean
from infinicore.tensor import (
    Tensor,
    empty,
    empty_like,
    from_blob,
    from_list,
    from_numpy,
    from_torch,
    ones,
    strided_empty,
    strided_from_blob,
    zeros,
)

__all__ = [
    # Modules.
    "context",
    "nn",
    # Classes.
    "device",
    "DeviceEvent",
    "dtype",
    "Tensor",
    # Context functions.
    "get_device",
    "get_device_count",
    "get_stream",
    "set_device",
    "sync_device",
    "sync_stream",
    "is_graph_recording",
    "start_graph_recording",
    "stop_graph_recording",
    # Data Types.
    "bfloat16",
    "bool",
    "cdouble",
    "cfloat",
    "chalf",
    "complex32",
    "complex64",
    "complex128",
    "double",
    "float",
    "float16",
    "float32",
    "float64",
    "half",
    "int",
    "int8",
    "int16",
    "int32",
    "int64",
    "long",
    "short",
    "uint8",
    # Operations.
    "addcmul",
    "atanh",
    "binary_cross_entropy_with_logits",
    "cdist",
    "reciprocal",
    "round",
    "add",
    "addr",
    "add_rms_norm",
    "add_rms_norm_",
    "argwhere",
    "asin",
    "attention",
    "kv_caching",
    "asinh",
    "baddbmm",
    "bilinear",
    "fmod",
    "cat",
    "inner",
    "masked_select",
    "matmul",
    "equal",
    "mul",
    "narrow",
    "squeeze",
    "unsqueeze",
    "rearrange",
    "cosh",
    "cross_entropy",
    "tan",
    "empty",
    "empty_like",
    "from_blob",
    "from_list",
    "from_numpy",
    "from_torch",
    "mha_kvcache",
    "mha_varlen",
    "fmin",
    "paged_caching",
    "paged_attention",
    "paged_attention_prefill",
    "hypot",
    "index_copy",
    "index_add",
    "take",
    "ones",
    "strided_empty",
    "strided_from_blob",
    "zeros",
    "sum",
    "var_mean",
    "var",
    "topk",
    "all",
]

use_ntops = False

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import sys

    import ntops

    for op_name in ntops.torch.__all__:
        getattr(ntops.torch, op_name).__globals__["torch"] = sys.modules[__name__]

    use_ntops = True
