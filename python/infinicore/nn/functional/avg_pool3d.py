import ctypes
from ctypes import c_size_t

import torch

from ...tensor import empty
from ...ops._infiniop_runtime import (
    _check_error,
    _load_lib,
    create_tensor_descriptor,
    destroy_tensor_descriptor,
    handle_for_tensor,
    infiniopOperatorDescriptor_t,
)


def _triple(v):
    if isinstance(v, int):
        return (v, v, v)
    if isinstance(v, (tuple, list)) and len(v) == 3:
        return (int(v[0]), int(v[1]), int(v[2]))
    raise ValueError(f"Expected int or tuple/list of len=3, got: {v!r}")


def _out_dim(in_size: int, kernel: int, stride: int, pad: int) -> int:
    return (in_size + 2 * pad - kernel) // stride + 1


def avg_pool3d(input, *, kernel_size, stride=None, padding=0):
    ks = _triple(kernel_size)
    st = None if stride is None else _triple(stride)
    pad = _triple(padding)

    # Follow torch.nn.functional.avg_pool3d default behavior (ceil_mode=False).
    n, c, d, h, w = input.shape
    sd, sh, sw = ks if st is None else st
    od = _out_dim(int(d), int(ks[0]), int(sd), int(pad[0]))
    oh = _out_dim(int(h), int(ks[1]), int(sh), int(pad[1]))
    ow = _out_dim(int(w), int(ks[2]), int(sw), int(pad[2]))
    out = empty([int(n), int(c), int(od), int(oh), int(ow)], dtype=input.dtype, device=input.device)

    lib = _load_lib()
    handle = handle_for_tensor(out)

    y_desc = create_tensor_descriptor(out)
    x_desc = create_tensor_descriptor(input)

    op_desc = infiniopOperatorDescriptor_t()

    ks_arr = (c_size_t * 3)(*ks)
    stride_ptr = None
    if st is not None:
        st_arr = (c_size_t * 3)(*st)
        stride_ptr = ctypes.cast(st_arr, ctypes.c_void_p)
    pad_arr = (c_size_t * 3)(*pad)

    _check_error(
        lib.infiniopCreateAvgPool3dDescriptor(
            handle,
            ctypes.byref(op_desc),
            y_desc,
            x_desc,
            ctypes.cast(ks_arr, ctypes.c_void_p),
            stride_ptr,
            ctypes.cast(pad_arr, ctypes.c_void_p),
        )
    )

    try:
        workspace_size = c_size_t(0)
        _check_error(
            lib.infiniopGetAvgPool3dWorkspaceSize(op_desc, ctypes.byref(workspace_size))
        )

        workspace = None
        workspace_ptr = None
        if workspace_size.value:
            workspace = torch.empty(
                (workspace_size.value,),
                dtype=torch.uint8,
                device=torch.device(str(out.device)),
            )
            workspace_ptr = ctypes.c_void_p(workspace.data_ptr())

        _check_error(
            lib.infiniopAvgPool3d(
                op_desc,
                workspace_ptr,
                c_size_t(workspace_size.value),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_void_p(input.data_ptr()),
                None,
            )
        )
        return out
    finally:
        _check_error(lib.infiniopDestroyAvgPool3dDescriptor(op_desc))
        destroy_tensor_descriptor(x_desc)
        destroy_tensor_descriptor(y_desc)

