from ctypes import POINTER, byref, c_int, c_int32, c_size_t, c_uint64, c_void_p

from infini.core import (
    infiniopHandle_t,
    infiniopOperatorDescriptor_t,
    infiniopTensorDescriptor_t,
    lib,
)

lib.infiniopCreateAddDescriptor.restype = c_int32
lib.infiniopCreateAddDescriptor.argtypes = [
    infiniopHandle_t,
    POINTER(infiniopOperatorDescriptor_t),
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
    infiniopTensorDescriptor_t,
]

lib.infiniopGetAddWorkspaceSize.restype = c_int32
lib.infiniopGetAddWorkspaceSize.argtypes = [
    infiniopOperatorDescriptor_t,
    POINTER(c_size_t),
]

lib.infiniopAdd.restype = c_int32
lib.infiniopAdd.argtypes = [
    infiniopOperatorDescriptor_t,
    c_void_p,
    c_size_t,
    c_void_p,
    c_void_p,
    c_void_p,
    c_void_p,
]

lib.infiniopDestroyAddDescriptor.restype = c_int32
lib.infiniopDestroyAddDescriptor.argtypes = [infiniopOperatorDescriptor_t]


def add(input, other, *, alpha=1, out=None):
    # if out is None:
    #     out = infini.empty_like(input)

    lib.infinirtSetDevice(lib.INFINI_DEVICE_NVIDIA, c_int(0))

    handle = infiniopHandle_t()
    lib.infiniopCreateHandle(byref(handle))

    descriptor = infiniopOperatorDescriptor_t()

    lib.infiniopCreateAddDescriptor(
        handle, byref(descriptor), out.descriptor, input.descriptor, other.descriptor
    )

    workspace_size = c_uint64(0)
    lib.infiniopGetAddWorkspaceSize(descriptor, byref(workspace_size))

    workspace = TestWorkspace(workspace_size.value, out.device)
    

    lib.infiniopAdd(
        descriptor,
        workspace.data(),
        workspace.size(),
        out.data(),
        input.data(),
        other.data(),
        None,
    )

    return out
