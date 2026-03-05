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
from ..tensor import empty


def dot(a, b):
    # Torch returns a scalar tensor for dot; represent it as a 0-d tensor.
    out_1d = empty([1], dtype=a.dtype, device=a.device)

    lib = _load_lib()
    handle = handle_for_tensor(out_1d)

    y_desc = create_tensor_descriptor(out_1d)
    a_desc = create_tensor_descriptor(a)
    b_desc = create_tensor_descriptor(b)

    op_desc = infiniopOperatorDescriptor_t()
    _check_error(
        lib.infiniopCreateDotDescriptor(
            handle, ctypes.byref(op_desc), y_desc, a_desc, b_desc
        )
    )

    try:
        workspace_size = c_size_t(0)
        _check_error(lib.infiniopGetDotWorkspaceSize(op_desc, ctypes.byref(workspace_size)))

        workspace = None
        workspace_ptr = None
        if workspace_size.value:
            workspace = torch.empty(
                (workspace_size.value,),
                dtype=torch.uint8,
                device=torch.device(str(out_1d.device)),
            )
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
        return out_1d.view([])
    finally:
        _check_error(lib.infiniopDestroyDotDescriptor(op_desc))
        destroy_tensor_descriptor(b_desc)
        destroy_tensor_descriptor(a_desc)
        destroy_tensor_descriptor(y_desc)

