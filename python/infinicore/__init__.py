import contextlib

import infinicore.nn as nn
from infinicore.device import device
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
from infinicore.ops.attention import attention
from infinicore.ops.matmul import matmul
from infinicore.ops.rearrange import rearrange
from infinicore.tensor import (
    Tensor,
    empty,
    empty_like,
    from_blob,
    ones,
    strided_empty,
    strided_from_blob,
    zeros,
)

__all__ = [
    # Modules.
    "nn",
    # Classes.
    "device",
    "dtype",
    "Tensor",
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
    "attention",
    "matmul",
    "rearrange",
    "empty",
    "empty_like",
    "from_blob",
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
