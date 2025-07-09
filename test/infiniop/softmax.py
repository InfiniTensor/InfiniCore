import torch
import ctypes
from ctypes import c_uint64, c_int32
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, reduce_axis, stride
    ((32, 20, 512), 0, (20 * 512, 512, 1)),
    ((32, 20, 512), 1, (20 * 512, 512, 1)),
    ((32, 20, 512), 2, (20 * 512, 512, 1)),
    
    # 2D 张量测试
    ((128, 256), 0, (256, 1)),
    ((128, 256), 1, (256, 1)),
    ((1024, 1024), 0, (1024, 1)),
    ((1024, 1024), 1, (1024, 1)),
    
    # 4D 张量测试
    ((8, 32, 64, 64), 0, (32 * 64 * 64, 64 * 64, 64, 1)),
    ((8, 32, 64, 64), 1, (32 * 64 * 64, 64 * 64, 64, 1)),
    ((8, 32, 64, 64), 2, (32 * 64 * 64, 64 * 64, 64, 1)),
    ((8, 32, 64, 64), 3, (32 * 64 * 64, 64 * 64, 64, 1)),
    
    # 5D 张量测试
    ((4, 16, 8, 32, 32), 0, (16 * 8 * 32 * 32, 8 * 32 * 32, 32 * 32, 32, 1)),
    ((4, 16, 8, 32, 32), 1, (16 * 8 * 32 * 32, 8 * 32 * 32, 32 * 32, 32, 1)),
    ((4, 16, 8, 32, 32), 2, (16 * 8 * 32 * 32, 8 * 32 * 32, 32 * 32, 32, 1)),
    ((4, 16, 8, 32, 32), 3, (16 * 8 * 32 * 32, 8 * 32 * 32, 32 * 32, 32, 1)),
    ((4, 16, 8, 32, 32), 4, (16 * 8 * 32 * 32, 8 * 32 * 32, 32 * 32, 32, 1)),
    
    # 小尺寸测试
    ((2, 3), 0, (3, 1)),
    ((2, 3), 1, (3, 1)),
    ((1, 10), 0, (10, 1)),
    ((1, 10), 1, (10, 1)),
    ((10, 1), 0, (1, 1)),
    ((10, 1), 1, (1, 1)),
    
    ((1000,), 0, (1,)),
    
    ((7, 333, 777), 0, (333 * 777, 777, 1)),
    ((7, 333, 777), 1, (333 * 777, 777, 1)),
    ((7, 333, 777), 2, (333 * 777, 777, 1)),
    ((13, 509, 251), 0, (509 * 251, 251, 1)),
    ((13, 509, 251), 1, (509 * 251, 251, 1)),
    ((13, 509, 251), 2, (509 * 251, 251, 1)),
    
    ((64, 1024, 768), 0, (1024 * 768, 768, 1)),
    ((64, 1024, 768), 1, (1024 * 768, 768, 1)),
    ((64, 1024, 768), 2, (1024 * 768, 768, 1)),
    ((32, 2048, 512), 0, (2048 * 512, 512, 1)),
    ((32, 2048, 512), 1, (2048 * 512, 512, 1)),
    ((32, 2048, 512), 2, (2048 * 512, 512, 1)),
    
    ((1024, 1), 0, (1, 1)),
    ((1024, 1), 1, (1, 1)),
    ((1, 1024), 0, (1024, 1)),
    ((1, 1024), 1, (1024, 1)),
    
    ((32, 8, 512, 64), 1, (8 * 512 * 64, 512 * 64, 64, 1)),
    ((32, 8, 512, 64), 2, (8 * 512 * 64, 512 * 64, 64, 1)),
    ((32, 8, 512, 64), 3, (8 * 512 * 64, 512 * 64, 64, 1)),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

def softmax(x, axis, y):
    torch.softmax(x, axis = axis, out=y)

def test(
    handle,
    device,
    shape,
    axis,
    stride,
    dtype=torch.float16,
    sync=None,
):
    x = TestTensor(shape, stride, dtype, device)
    y = TestTensor(shape, stride, dtype, device, mode="zeros")

    print(
        f"Testing softmax on {InfiniDeviceNames[device]} with shape:{shape} stride:{stride} axis:{axis} "
        f"dtype:{dtype}"
    )
    # a = torch.randn(shape, dtype=dtype).to(torch_device) * 0.1
    # b = torch.empty_like(a)
    softmax(x.torch_tensor(), axis, y.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSoftmaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            c_int32(axis),
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSoftmaxWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_softmax():
        check_error(
            LIBINFINIOP.infiniopSoftmax(
                descriptor,
                workspace.data() if workspace is not None else None,
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )

    lib_softmax()

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, y]:
        tensor.destroy_desc()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: softmax(x.torch_tensor(), axis, y.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_softmax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroySoftmaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    
    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")

