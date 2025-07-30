import ctypes
from ctypes import c_uint64
from enum import Enum, auto
import math

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # tensor_shape, inplace
    # TODO: Uncomment the following line.
    # ((),),
    ((1, 3),),
    ((3, 3),),
    ((32, 20, 512),),
    ((33, 333, 333),),
    ((32, 256, 112, 112),),
    ((3, 3, 13, 9, 17),),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
# _TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]
# _TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]
_TENSOR_DTYPES = [InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def gelu_backward(input, grad_output):
    sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
    alpha = 0.044715

    x = input
    x_cubed = x * x * x
    inner = sqrt_2_over_pi * (x + alpha * x_cubed)
    tanh_inner = torch.tanh(inner)

    # derivative of tanh part
    sech_squared = 1 - tanh_inner * tanh_inner
    left = 0.5 * (1 + tanh_inner)
    right = 0.5 * x * sech_squared * sqrt_2_over_pi * (1 + 3 * alpha * x * x)

    grad_input = grad_output * (left + right)
    return grad_input


def test(
    handle, device, shape, inplace=Inplace.OUT_OF_PLACE, dtype=torch.float16, sync=None
):
    x_torch_tensor = torch.rand(shape) * 2 - 1

    input = TestTensor(
        shape,
        x_torch_tensor.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=x_torch_tensor,
    )
    grad_output_torch_tensor = torch.rand(shape) * 4 - 2
    grad_output = TestTensor(
        shape,
        grad_output_torch_tensor.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=x_torch_tensor,
    )

    if inplace == Inplace.INPLACE_X:
        y = input
    else:
        y = TestTensor(shape, None, dtype, device)

    if y.is_broadcast():
        return

    print(
        f"Testing gelu_backward on {InfiniDeviceNames[device]} with shape:{shape} dtype:{InfiniDtypeNames[dtype]} inplace: {inplace}"
    )

    ans = gelu_backward(input.torch_tensor(), grad_output.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGeluBackWardDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            input.descriptor,
            grad_output.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y, input, grad_output]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGeluBackWardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_gelu_backward():
        LIBINFINIOP.infiniopGeluBackWard(
            descriptor,
            workspace.data(),
            workspace.size(),
            y.data(),
            input.data(),
            grad_output.data(),
            None,
        )

    lib_gelu_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    # print(y.actual_tensor())
    # print(ans)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: gelu_backward(input.torch_tensor(), grad_output.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gelu_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyGeluBackWardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
