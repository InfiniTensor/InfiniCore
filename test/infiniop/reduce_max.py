import torch
import ctypes
from ctypes import c_uint64
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

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, dim
    ((4, 8), 0),
    ((4, 8), 1),
    ((2, 3, 4), 0),
    ((2, 3, 4), 1),
    ((2, 3, 4), 2),
    ((16, 32, 64), 0),
    ((16, 32, 64), 1),
    ((16, 32, 64), 2),
    ((8, 16, 32, 64), 0),
    ((8, 16, 32, 64), 1),
    ((8, 16, 32, 64), 2),
    ((8, 16, 32, 64), 3),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def reduce_max(input_tensor, dim):
    """PyTorch reference implementation for ReduceMax"""
    return torch.max(input_tensor, dim=dim, keepdim=True)[0]


def test(
    handle,
    device,
    shape,
    dim,
    dtype=torch.float16,
    sync=None,
):
    """Test function for ReduceMax operator"""
    print(
        f"Testing ReduceMax on {InfiniDeviceNames[device]} with shape:{shape} dim:{dim} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    
    # Calculate output shape (reduce dimension but keep it as 1)
    output_shape = list(shape)
    output_shape[dim] = 1
    
    # Create input torch tensor with random values
    input_torch_tensor = torch.rand(shape) * 2 - 1
    
    # Create input tensor
    input_tensor = TestTensor(
        shape,
        input_torch_tensor.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=input_torch_tensor,
    )
    
    # Create output tensor
    output_tensor = TestTensor(output_shape, None, dtype, device)
    
    # Compute PyTorch reference result
    output_tensor.torch_tensor().copy_(reduce_max(input_tensor.torch_tensor(), dim))
    
    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReduceMaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
            ctypes.c_int32(dim),
        )
    )
    
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, output_tensor]:
        tensor.destroy_desc()
    
    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReduceMaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    
    # Create workspace
    workspace = TestWorkspace(workspace_size.value, device)
    
    def lib_reduce_max():
        check_error(
            LIBINFINIOP.infiniopReduceMax(
                descriptor,
                workspace.data(),
                workspace.size(),
                output_tensor.data(),
                input_tensor.data(),
                None,
            )
        )
    
    # Execute the operation
    lib_reduce_max()
    
    if sync is not None:
        sync()
    
    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output_tensor.actual_tensor(), output_tensor.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(output_tensor.actual_tensor(), output_tensor.torch_tensor(), atol=atol, rtol=rtol)
    
    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: reduce_max(input_tensor.torch_tensor(), dim), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_reduce_max(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    # Clean up
    check_error(LIBINFINIOP.infiniopDestroyReduceMaxDescriptor(descriptor))


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