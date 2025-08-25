import torch
import ctypes
from ctypes import c_uint64, c_float
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
import torch.nn.functional as F

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_INPUT = auto()

# Test cases: (shape, input_stride, weight_stride, bias_stride, running_mean_stride, running_var_stride, output_stride)
_TEST_CASES_ = [
    # 3D BatchNorm1d cases - 符合competition.md要求的3维 (Batch, Channel, Dim)
    ((2, 16, 32), None, None, None, None, None, None),       # Small 3D case
    ((4, 64, 128), None, None, None, None, None, None),      # Medium 3D case
    ((8, 128, 256), None, None, None, None, None, None),     # Large 3D case
    ((1, 512, 1024), None, None, None, None, None, None),    # Single batch case
    ((16, 32, 64), None, None, None, None, None, None),      # Batch processing
]

# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_INPUT,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_EXPANDED_TEST_CASES_ = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]  # 支持F32, F16, BF16

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},  # BF16 needs higher tolerance
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def batch_norm(handle, output, input_tensor, weight, bias, running_mean, running_var, momentum=0.1, eps=1e-5):
    """Call the InfiniOp BatchNorm implementation"""
    # Create descriptor
    desc = infiniopOperatorDescriptor_t()
    
    # Create BatchNorm descriptor
    check_error(
        LIBINFINIOP.infiniopCreateBatchNormDescriptor(
            handle,
            ctypes.byref(desc),
            output.descriptor,
            input_tensor.descriptor,
            weight.descriptor,
            bias.descriptor,
            running_mean.descriptor,
            running_var.descriptor,
            c_float(momentum),
            c_float(eps),
        )
    )
    
    # Get workspace size
    workspace_size = c_uint64()
    check_error(
        LIBINFINIOP.infiniopGetBatchNormWorkspaceSize(
            desc, ctypes.byref(workspace_size)
        )
    )
    
    # Create workspace
    workspace = TestWorkspace(workspace_size.value, input_tensor.device)
    
    # Execute BatchNorm
    check_error(
        LIBINFINIOP.infiniopBatchNorm(
            desc,
            workspace.data(),
            workspace.size(),
            output.data(),
            input_tensor.data(),
            weight.data(),
            bias.data(),
            running_mean.data(),
            running_var.data(),
            None,  # stream
        )
    )
    
    # Destroy descriptor
    check_error(LIBINFINIOP.infiniopDestroyBatchNormDescriptor(desc))


def test(
    handle,
    device,
    input_shape,
    input_stride=None,
    weight_stride=None,
    bias_stride=None,
    running_mean_stride=None,
    running_var_stride=None,
    output_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    tensor_dtype=InfiniDtype.F32,
    sync=None,
):
    """Test function for BatchNorm operator"""
    inplace_str = "inplace" if inplace == Inplace.INPLACE_INPUT else "out-of-place"
    print(
        f"Testing BatchNorm on {InfiniDeviceNames[device]} with shape:{input_shape} "
        f"dtype:{InfiniDtypeNames[tensor_dtype]} mode:{inplace_str}"
    )

    
    # Create input tensors
    input_tensor = TestTensor(
        input_shape, input_stride, tensor_dtype, device, mode="random", scale=2.0, bias=-1.0
    )
    
    # BatchNorm parameters (1D tensors with length = channels)
    channels = input_shape[1]
    param_shape = (channels,)
    
    weight = TestTensor(
        param_shape, weight_stride, tensor_dtype, device, mode="ones"
    )
    bias = TestTensor(
        param_shape, bias_stride, tensor_dtype, device, mode="zeros"
    )
    running_mean = TestTensor(
        param_shape, running_mean_stride, tensor_dtype, device, mode="zeros"
    )
    running_var = TestTensor(
        param_shape, running_var_stride, tensor_dtype, device, mode="ones"
    )
    
    # Create output tensor - for inplace, reuse input tensor
    if inplace == Inplace.INPLACE_INPUT:
        output = input_tensor
    else:
        output = TestTensor(
            input_shape, output_stride, tensor_dtype, device, mode="zeros"
        )
    
    # Set momentum and eps
    momentum = 0.1
    eps = 1e-5
    
    # Execute InfiniOp BatchNorm
    if PROFILE:
        profile_operation(
            lambda: batch_norm(
                handle, output, input_tensor, weight, bias, running_mean, running_var, momentum, eps
            ),
            f"InfiniOp BatchNorm {InfiniDtypeNames[tensor_dtype]} {input_shape}",
            NUM_PRERUN,
            NUM_ITERATIONS,
            sync,
        )
    else:
        batch_norm(handle, output, input_tensor, weight, bias, running_mean, running_var, momentum, eps)
    
    # Convert to PyTorch tensors for reference computation
    # For inplace, we need to save the original input for comparison
    if inplace == Inplace.INPLACE_INPUT:
        input_torch = input_tensor.torch_tensor().clone()  # Save original for reference
    else:
        input_torch = input_tensor.torch_tensor()
    
    weight_torch = weight.torch_tensor()
    bias_torch = bias.torch_tensor()
    running_mean_torch = running_mean.torch_tensor()
    running_var_torch = running_var.torch_tensor().clone()
    
    # PyTorch BatchNorm1d for 3D tensors (Batch, Channel, Dim) - 符合competition.md要求
    if len(input_shape) == 3:
        torch_bn = torch.nn.BatchNorm1d(channels, momentum=momentum, eps=eps, affine=True, track_running_stats=True)
    else:
        raise ValueError(f"Only 3D tensors are supported according to competition.md, got {len(input_shape)}D")
    
    torch_bn.weight.data = weight_torch.clone()
    torch_bn.bias.data = bias_torch.clone()
    torch_bn.running_mean.data = running_mean_torch.clone()
    torch_bn.running_var.data = running_var_torch.clone()
    torch_bn.train()  # Set to training mode
    
    # PyTorch reference computation
    if PROFILE:
        def pytorch_batch_norm():
            return torch_bn(input_torch)
        
        expected = profile_operation(
            pytorch_batch_norm,
            f"PyTorch BatchNorm {InfiniDtypeNames[tensor_dtype]} {input_shape}",
            NUM_PRERUN,
            NUM_ITERATIONS,
            sync,
        )
    else:
        expected = torch_bn(input_torch)
    

    
    # Compare results
    actual = output.actual_tensor()
    
    # Check correctness
    tolerance = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    atol, rtol = tolerance
    
    if DEBUG:
        debug(actual, expected.detach(), atol=atol, rtol=rtol)
    torch.testing.assert_close(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
        msg=f"BatchNorm test failed for shape {input_shape}, dtype {InfiniDtypeNames[tensor_dtype]}"
    )
    
    # Clean up
    input_tensor.destroy_desc()
    weight.destroy_desc()
    bias.destroy_desc()
    running_mean.destroy_desc()
    running_var.destroy_desc()
    
    # For inplace, output is the same as input_tensor, so don't destroy twice
    if inplace != Inplace.INPLACE_INPUT:
        output.destroy_desc()


if __name__ == "__main__":
    args = get_args()
    
    # Update global settings
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    for device in get_test_devices(args):
        test_operator(device, test, _EXPANDED_TEST_CASES_, _TENSOR_DTYPES)
    
    print("\033[92mBatchNorm test passed!\033[0m")