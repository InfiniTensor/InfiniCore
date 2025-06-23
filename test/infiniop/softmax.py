import torch
import ctypes
from ctypes import POINTER, Structure, c_int32, c_void_p, c_uint64
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    create_workspace,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape  reduce_axis
    ((32, 20, 512), 0),
    ((32, 20, 512), 1), 
    ((32, 20, 512), 2),
    
    # 2D 张量测试
    ((128, 256), 0),
    ((128, 256), 1),
    ((1024, 1024), 0),
    ((1024, 1024), 1),
    
    # 4D 张量测试
    ((8, 32, 64, 64), 0),    
    ((8, 32, 64, 64), 1),    
    ((8, 32, 64, 64), 2),    
    ((8, 32, 64, 64), 3),   
    
    # 5D 张量测试
    ((4, 16, 8, 32, 32), 0),
    ((4, 16, 8, 32, 32), 1),
    ((4, 16, 8, 32, 32), 2),
    ((4, 16, 8, 32, 32), 3),
    ((4, 16, 8, 32, 32), 4),
    
    # 小尺寸测试
    ((2, 3), 0),
    ((2, 3), 1),
    ((1, 10), 0),
    ((1, 10), 1),
    ((10, 1), 0),
    ((10, 1), 1),
    
    ((1000,), 0),
    
    ((7, 333, 777), 0),
    ((7, 333, 777), 1),
    ((7, 333, 777), 2),
    ((13, 509, 251), 0),
    ((13, 509, 251), 1),
    ((13, 509, 251), 2),
    
    ((64, 1024, 768), 0),   
    ((64, 1024, 768), 1),
    ((64, 1024, 768), 2),
    ((32, 2048, 512), 0),   
    ((32, 2048, 512), 1),
    ((32, 2048, 512), 2),

    ((1024, 1), 0),          
    ((1024, 1), 1),
    ((1, 1024), 0),        
    ((1, 1024), 1),

    ((32, 8, 512, 64), 1),   
    ((32, 8, 512, 64), 2),   
    ((32, 8, 512, 64), 3),   
]


# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.float32: {"atol": 1e-7, "rtol": 1e-7},
}


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class SoftmaxDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopSoftmaxDescriptor_t = POINTER(SoftmaxDescriptor)


def softmax(x, axis):
    return torch.softmax(x, axis = axis).to(x.dtype)

def test(
    lib,
    handle,
    torch_device,
    shape,
    axis,
    dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing softmax on {torch_device} with shape:{shape}"
        f"dtype:{dtype}"
    )

    a = torch.randn(shape, dtype=dtype).to(torch_device) * 0.1
    b = torch.empty_like(a)
    ans = softmax(a, axis)

    a_tensor, b_tensor = [to_tensor(tensor, lib) for tensor in [a, b]]

    if sync is not None:
        sync()

    descriptor = infiniopSoftmaxDescriptor_t()
    check_error(
        lib.infiniopCreateSoftmaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            b_tensor.descriptor,
            a_tensor.descriptor,
            c_int32(axis),
        )
    )


    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a_tensor, b_tensor]:
        tensor.destroyDesc(lib)

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetSoftmaxWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, a.device)

    def lib_softmax():
        check_error(
            lib.infiniopSoftmax(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                b_tensor.data,
                a_tensor.data,
                None,
            )
        )

    lib_softmax()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(b, ans, atol=atol, rtol=rtol)
    assert torch.allclose(b, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: softmax(a, b), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_softmax(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroySoftmaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()

    lib.infiniopCreateSoftmaxDescriptor.restype = c_int32
    lib.infiniopCreateSoftmaxDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopSoftmaxDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_int32,
    ]

    lib.infiniopGetSoftmaxWorkspaceSize.restype = c_int32
    lib.infiniopGetSoftmaxWorkspaceSize.argtypes = [
        infiniopSoftmaxDescriptor_t,
        POINTER(c_uint64),
    ]

    lib.infiniopSoftmax.restype = c_int32
    lib.infiniopSoftmax.argtypes = [
        infiniopSoftmaxDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
    ]

    lib.infiniopDestroySoftmaxDescriptor.restype = c_int32
    lib.infiniopDestroySoftmaxDescriptor.argtypes = [
        infiniopSoftmaxDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")

