import ctypes
from ctypes import c_size_t

from ._infiniop_runtime import (
    _check_error,
    _load_lib,
    create_tensor_descriptor,
    destroy_tensor_descriptor,
    handle_for_tensor,
    infiniopOperatorDescriptor_t,
)
from ..dtype import uint8
from ..tensor import empty
from ..tensor import empty_like


def log1p(input, *, out=None):
    if out is None:
        out = empty_like(input)

    lib = _load_lib()
    handle = handle_for_tensor(out)

    op_desc = infiniopOperatorDescriptor_t()
    y_desc = None
    x_desc = None

    try:
        y_desc = create_tensor_descriptor(out)
        x_desc = create_tensor_descriptor(input)
        _check_error(
            lib.infiniopCreateLog1pDescriptor(
                handle, ctypes.byref(op_desc), y_desc, x_desc
            )
        )

        workspace_size = c_size_t(0)
        _check_error(
            lib.infiniopGetLog1pWorkspaceSize(op_desc, ctypes.byref(workspace_size))
        )

        workspace = None
        workspace_ptr = None
        if workspace_size.value:
            workspace = empty([int(workspace_size.value)], dtype=uint8, device=out.device)
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
        if workspace is not None:
            keepalive = getattr(out, "_infiniop_keepalive", None)
            if keepalive is None:
                keepalive = []
                out._infiniop_keepalive = keepalive
            keepalive.append(workspace)
        return out
    finally:
        if op_desc:
            _check_error(lib.infiniopDestroyLog1pDescriptor(op_desc))
        if x_desc:
            destroy_tensor_descriptor(x_desc)
        if y_desc:
            destroy_tensor_descriptor(y_desc)
