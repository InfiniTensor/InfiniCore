import ctypes
from ctypes import c_size_t

import torch

from ._infiniop_runtime import (
    _check_error,
    _load_lib,
    create_tensor_descriptor,
    destroy_tensor_descriptor,
    handle_for_tensor,
    infiniopOperatorDescriptor_t,
)
from ..tensor import empty_like


def log1p(input, *, out=None):
    if out is None:
        out = empty_like(input)

    lib = _load_lib()
    handle = handle_for_tensor(out)

    y_desc = create_tensor_descriptor(out)
    x_desc = create_tensor_descriptor(input)

    op_desc = infiniopOperatorDescriptor_t()
    _check_error(
        lib.infiniopCreateLog1pDescriptor(
            handle, ctypes.byref(op_desc), y_desc, x_desc
        )
    )

    try:
        workspace_size = c_size_t(0)
        _check_error(
            lib.infiniopGetLog1pWorkspaceSize(op_desc, ctypes.byref(workspace_size))
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
            lib.infiniopLog1p(
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
        _check_error(lib.infiniopDestroyLog1pDescriptor(op_desc))
        destroy_tensor_descriptor(x_desc)
        destroy_tensor_descriptor(y_desc)

