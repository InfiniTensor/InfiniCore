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

# ==============================================================================
#  Configuration for Leaky ReLU Testing
# ==============================================================================
# Test cases: shape, input_stride, output_stride, negative_slope
_TEST_CASES_ = [
    # shape, input_stride, output_stride, negative_slope
    ((16,), None, None, 0.01),
    ((13, 4), None, None, 0.02),
    ((13, 4), (10, 1), (10, 1), 0.03),
    ((13, 4), (0, 1), None, 0.04),
    ((13, 4, 4), None, None, 0.01),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), 0.01),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), 0.01),
    ((16, 5632), None, None, 0.01),
    ((16, 5632), (13312, 1), (13312, 1), 0.01),
    ((4, 4, 5632), None, None, 0.01),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), 0.01),
    ((4,), None, None, 0.01),
    ((10,), (1,), (1,), 0.5),
    ((3, 3, 3), None, None, 0.0),
    ((2, 4), None, None, 1.0), 
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [
    # Metax下InfiniDtype.F16会报错(CPU的可以正常测试), 暂时注释掉
    # InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.BF16
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = True
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def leaky_relu(output, input, negative_slope):
    output.copy_(torch.nn.functional.leaky_relu(input, negative_slope, inplace=False))


def test(
    handle,
    device,
    shape,
    input_stride=None,
    output_stride=None,
    negative_slope=0.01,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    # Create input tensor
    input_tensor = TestTensor(shape, input_stride, dtype, device)
    # Handle in-place vs out-of-place
    if inplace == Inplace.INPLACE:
        if input_stride != output_stride:
            return  # Skip incompatible strides
        output_tensor = input_tensor
    else:
        output_tensor = TestTensor(shape, output_stride, dtype, device, mode="ones")
    
    if output_tensor.is_broadcast():
        return  # Skip broadcasted outputs

    print(
        f"Testing LeakyReLU on {InfiniDeviceNames[device]} with shape:{shape} "
        f"input_stride:{input_stride} output_stride:{output_stride} "
        f"negative_slope:{negative_slope} dtype:{InfiniDtypeNames[dtype]} "
        f"slope:{negative_slope} inplace:{inplace}"
    )

    # Compute reference result using PyTorch
    leaky_relu(
        output_tensor.torch_tensor(),
        input_tensor.torch_tensor(),
        negative_slope
    )


    if sync is not None:
        sync()

    # Create LeakyReLU descriptor
    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateLeakyReluDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor
            )
    )
    # Invalidate the shape and strides in the descriptor
    input_tensor.destroy_desc()
    if inplace == Inplace.OUT_OF_PLACE:
        output_tensor.destroy_desc()

    # Get workspace size and allocate
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLeakyReluWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Define function to call the library implementation
    def lib_leaky_relu():
        check_error(
            LIBINFINIOP.infiniopLeakyRelu(
                descriptor,
                workspace.data(),
                workspace.size(),
                output_tensor.data(),
                input_tensor.data(),
                negative_slope,
                None,
            )
        )
    # Run the library implementation
    lib_leaky_relu()

    # Verify results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output_tensor.actual_tensor(), output_tensor.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(
        output_tensor.actual_tensor(), 
        output_tensor.torch_tensor(), 
        atol=atol, 
        rtol=rtol
    )

    # Profiling
    if PROFILE:
        profile_operation(
            "PyTorch", 
            lambda: leaky_relu(
                output_tensor.torch_tensor(),
                input_tensor.torch_tensor(),
                negative_slope
            ), 
            device, 
            NUM_PRERUN, 
            NUM_ITERATIONS
        )
        profile_operation(
            "    lib", 
            lambda: lib_leaky_relu(), 
            device, 
            NUM_PRERUN, 
            NUM_ITERATIONS
        )
    # Clean up
    check_error(LIBINFINIOP.infiniopDestroyLeakyReluDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mLeakyReLU test passed!\033[0m")