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
from enum import Enum, auto

_TEST_CASES_ = [
    # shape, grad_input_stride, grad_output_stride, input_stride
    ((13, 4), None, None, None),
    ((13, 4), (1, 13), (1, 13), (1, 13)),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (4, 1, 52), (16, 4, 1), (1, 13, 52)),
    ((16, 5632), None, None, None),
    ((16, 5632), (1, 16), (1, 16), (1, 16)),
    ((4, 4, 5632), None, None, None),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
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
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]
# _TENSOR_DTYPES = [InfiniDtype.F32]

# Tolerance map for different data types

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_gelu_backward(grad_input:torch.Tensor, input:torch.Tensor, grad_output:torch.Tensor):
    input_clone = input.clone()
    input_clone.requires_grad_(True)
    output = torch.nn.functional.gelu(input_clone, approximate="tanh")
    output.backward(grad_output)
    grad_input.copy_(input_clone.grad)
    

def test(
    handle,
    device,
    input_shape,
    grad_input_stride,
    grad_output_stride,
    input_stride,
    inplace,
    dtype,
    sync=None,
):
    print(
        f"Testing gelu_backward on {InfiniDeviceNames[device]} with input_shape:{input_shape},"
        f"grad_input_stride:{grad_input_stride},grad_output_stride:{grad_output_stride},input_stride:{input_stride},"
        f"inplace:{inplace}",
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    # positive and negative
    torch_input = torch.rand(input_shape) * 10 - 5
    if input_stride is not None:
        torch_input.as_strided_(input_shape, input_stride)
    input = TestTensor(
        input_shape,
        torch_input.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_input
    )
    grad_output = TestTensor(
        input_shape,
        grad_output_stride,
        dtype,
        device,
    )
    if inplace == Inplace.INPLACE:
        if grad_input_stride != grad_output_stride:
            return
        grad_input = grad_output
    else:
        grad_input = TestTensor(
            input_shape,
            grad_input_stride,
            dtype,
            device,
        )    

    torch_gelu_backward(grad_input.torch_tensor(), input.torch_tensor(), grad_output.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGeLUBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_input.descriptor,
            input.descriptor,
            grad_output.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [grad_input, input, grad_output]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGeLUBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_input.device)

    def lib_gelu_backward():
        check_error(
            LIBINFINIOP.infiniopGeLUBackward(
                descriptor,
                workspace.data(),
                workspace.size(),
                grad_input.data(),
                input.data(),
                grad_output.data(),                
                None,
            )
        )

    lib_gelu_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol)


    assert torch.allclose(grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_gelu_backward(
            grad_input.torch_tensor(), input.torch_tensor(), grad_output.torch_tensor()
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gelu_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGeLUBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my gelu_backward passed!\033[0m")
