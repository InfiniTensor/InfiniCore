import contextlib

import infinicore.context as context
import infinicore.nn as nn
from infinicore._device import device
from infinicore._dtype import (
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
from infinicore.device_event import DeviceEvent
from infinicore.ops.add import add
from infinicore.ops.add_rms_norm import add_rms_norm, add_rms_norm_
from infinicore.ops.attention import attention
from infinicore.ops.matmul import matmul
from infinicore.ops.mul import mul
from infinicore.ops.narrow import narrow
from infinicore.ops.paged_attention import paged_attention
from infinicore.ops.paged_attention_prefill import paged_attention_prefill
from infinicore.ops.paged_caching import paged_caching
from infinicore.ops.rearrange import rearrange
from infinicore.ops.squeeze import squeeze
from infinicore.ops.unsqueeze import unsqueeze
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
    "add",
    "add_rms_norm",
    "add_rms_norm_",
    "attention",
    "matmul",
    "mul",
    "narrow",
    "squeeze",
    "unsqueeze",
    "rearrange",
    "empty",
    "empty_like",
    "from_blob",
    "from_list",
    "from_numpy",
    "from_torch",
    "paged_caching",
    "paged_attention",
    "paged_attention_prefill",
    "ones",
    "strided_empty",
    "strided_from_blob",
    "zeros",
]

use_ntops = False

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import sys

    import ntops

    for op_name in ntops.torch.__all__:
        getattr(ntops.torch, op_name).__globals__["torch"] = sys.modules[__name__]

    use_ntops = True
