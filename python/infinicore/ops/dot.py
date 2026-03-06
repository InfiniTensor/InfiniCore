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


def dot(a, b):
    # Torch returns a scalar tensor for dot; represent it as a 0-d tensor.
    out_1d = empty([1], dtype=a.dtype, device=a.device)

    lib = _load_lib()
    handle = handle_for_tensor(out_1d)

    op_desc = infiniopOperatorDescriptor_t()
    y_desc = None
    a_desc = None
    b_desc = None

    try:
        y_desc = create_tensor_descriptor(out_1d)
        a_desc = create_tensor_descriptor(a)
        b_desc = create_tensor_descriptor(b)
        _check_error(
            lib.infiniopCreateDotDescriptor(
                handle, ctypes.byref(op_desc), y_desc, a_desc, b_desc
            )
        )

        workspace_size = c_size_t(0)
        _check_error(lib.infiniopGetDotWorkspaceSize(op_desc, ctypes.byref(workspace_size)))

        workspace = None
        workspace_ptr = None
        if workspace_size.value:
            workspace = empty([int(workspace_size.value)], dtype=uint8, device=out_1d.device)
            workspace_ptr = ctypes.c_void_p(workspace.data_ptr())

        _check_error(
            lib.infiniopDot(
                op_desc,
                workspace_ptr,
                c_size_t(workspace_size.value),
                ctypes.c_void_p(out_1d.data_ptr()),
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b.data_ptr()),
                None,
            )
        )
        out = out_1d.view([])
        if workspace is not None:
            keepalive = getattr(out, "_infiniop_keepalive", None)
            if keepalive is None:
                keepalive = []
                out._infiniop_keepalive = keepalive
            keepalive.append(workspace)
        return out
    finally:
        if op_desc:
            _check_error(lib.infiniopDestroyDotDescriptor(op_desc))
        if b_desc:
            destroy_tensor_descriptor(b_desc)
        if a_desc:
            destroy_tensor_descriptor(a_desc)
        if y_desc:
            destroy_tensor_descriptor(y_desc)
