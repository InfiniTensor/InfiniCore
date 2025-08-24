#!/usr/bin/env python3

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

# Test cases for reduce_mean
_TEST_CASES_ = [
    # (input_shape, reduce_dim)
    ((2, 2), 0),
    ((2, 2), 1),
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

_TENSOR_DTYPES = [
    InfiniDtype.F32,
    InfiniDtype.F16,
    InfiniDtype.BF16,
]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def reduce_mean(input_tensor, dim):
    """Reference implementation using PyTorch"""
    return torch.mean(input_tensor, dim=dim, keepdim=True)

def test(
    handle,
    device,
    shape,
    dim,
    dtype=torch.float16,
    sync=None,
):
    """Test function for ReduceMean operator"""
    print(
        f"Testing ReduceMean on {InfiniDeviceNames[device]} with shape:{shape} dim:{dim} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    
    # Calculate output shape (keep the reduced dimension as 1)
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
    pytorch_result = reduce_mean(input_tensor.torch_tensor(), dim)
    output_tensor.torch_tensor().copy_(pytorch_result)
    
    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReduceMeanDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
            ctypes.c_size_t(dim),
        )
    )
    
    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, output_tensor]:
        tensor.destroy_desc()
    
    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReduceMeanWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    
    # Create workspace
    workspace = TestWorkspace(workspace_size.value, device)
    
    def lib_reduce_mean():
        # Keep references to prevent garbage collection
        input_torch_tensor = input_tensor.torch_tensor()
        output_torch_tensor = output_tensor.torch_tensor()
        
        # Ensure tensors are contiguous
        input_torch_tensor = input_torch_tensor.contiguous()
        output_torch_tensor = output_torch_tensor.contiguous()
        
        # Get data pointers
        input_ptr = input_torch_tensor.data_ptr()
        output_ptr = output_torch_tensor.data_ptr()
        
        # Store reference to original output tensor
        original_output = output_tensor.torch_tensor()
        
        check_error(
            LIBINFINIOP.infiniopReduceMean(
                descriptor,
                workspace.data(),
                workspace.size(),
                ctypes.c_void_p(output_ptr),
                ctypes.c_void_p(input_ptr),
                None,
            )
        )
        
        # Copy results back to original tensor if contiguous tensor was created
        if output_torch_tensor.data_ptr() != original_output.data_ptr():
            original_output.copy_(output_torch_tensor)
        
        # Ensure _torch_tensor is updated with the C++ results
        output_tensor.torch_tensor().copy_(output_tensor.actual_tensor())
    
    # Execute the operation
    lib_reduce_mean()
    
    # Synchronize if needed
    if sync is not None:
        sync()
    
    # Compare results
    tolerance_dict = _TOLERANCE_MAP.get(dtype, {"atol": 1e-7, "rtol": 1e-7})
    
    if DEBUG:
        debug(
            output_tensor.actual_tensor(),
            output_tensor.torch_tensor(),
            **tolerance_dict,
        )
    
    if not torch.allclose(
        output_tensor.actual_tensor(), output_tensor.torch_tensor(), **tolerance_dict
    ):
        raise AssertionError("Results do not match")
    
    # Profile if needed
    if PROFILE:
        profile_operation(
            lib_reduce_mean,
            f"reduce_mean {InfiniDtypeNames[dtype]} {shape} dim={dim}",
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
    
    # Cleanup
    check_error(LIBINFINIOP.infiniopDestroyReduceMeanDescriptor(descriptor))


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