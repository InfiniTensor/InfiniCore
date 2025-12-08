import contextlib

import infinicore.context as context
import infinicore.nn as nn

# Import context functions
from infinicore.context import (
    get_device,
    get_device_count,
    get_stream,
    set_device,
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
from infinicore.ops.addbmm import addbmm
from infinicore.ops.attention import attention
from infinicore.ops.floor import floor
from infinicore.ops.hypot import hypot
from infinicore.ops.index_add import index_add
from infinicore.ops.index_copy import index_copy
from infinicore.ops.acos import acos
from infinicore.ops.matmul import matmul
from infinicore.ops.mul import mul
from infinicore.ops.narrow import narrow
from infinicore.ops.rearrange import rearrange
from infinicore.ops.take import take
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
    "acos",
    "add",
    "addbmm",
    "attention",
    "index_add",
    "index_copy",
    "matmul",
    "mul",
    "narrow",
    "rearrange",
    "empty",
    "empty_like",
    "floor",
    "from_blob",
    "from_list",
    "from_numpy",
    "from_torch",
    "hypot",
    "ones",
    "strided_empty",
    "strided_from_blob",
    "take",
    "zeros",
]

use_ntops = False

with contextlib.suppress(ImportError, ModuleNotFoundError):
    import sys

    import ntops

    for op_name in ntops.torch.__all__:
        getattr(ntops.torch, op_name).__globals__["torch"] = sys.modules[__name__]

    use_ntops = True
