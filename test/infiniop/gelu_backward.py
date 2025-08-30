import ctypes
from ctypes import c_uint64
from enum import Enum, auto

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceEnum,
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
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_A,
    Inplace.INPLACE_B,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.F64: {"atol": 1e-8, "rtol": 1e-8},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def gelu_backward(input: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi, device=input.device))
    kappa = 0.044715

    # 计算中间变量 u = √(2/π)(x + κx³)
    x_cubed = input.pow(3)
    u = sqrt_2_over_pi * (input + kappa * x_cubed)

    # 计算 tanh(u) 及其导数 sech²(u) = 1 - tanh²(u)
    tanh_u = torch.tanh(u)
    sech2_u = 1.0 - tanh_u.square()

    # 计算 du/dx = √(2/π)(1 + 3κx²)
    du_dx = sqrt_2_over_pi * (1.0 + 3 * kappa * input.square())

    # 局部梯度 dy/dx = 0.5*(1 + tanh_u) + 0.5*x*sech2_u*du_dx
    dy_dx = 0.5 * (1.0 + tanh_u) + 0.5 * input * sech2_u * du_dx

    return grad_output * dy_dx


def test(
    handle,
    device,
    shape,
    input_stride=None,
    grad_output_stride=None,
    grad_input_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    input = TestTensor(shape, input_stride, dtype, device)
    grad_output = TestTensor(shape, grad_output_stride, dtype, device)
    if inplace == Inplace.INPLACE_A:
        if input_stride != grad_input_stride:
            return
        grad_input = input
    elif inplace == Inplace.INPLACE_B:
        if grad_output_stride != grad_input_stride:
            return
        grad_input = grad_output
    else:
        grad_input = TestTensor(shape, grad_input_stride, dtype, device, mode="ones")

    if grad_input.is_broadcast():
        return

    print(
        f"Testing GeluBackward on {InfiniDeviceNames[device]} with shape:{shape} input_stride:{input_stride} grad_output_stride:{grad_output_stride} grad_input_stride:{grad_input_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    new_grad_input = gelu_backward(input.torch_tensor(), grad_output.torch_tensor())
    grad_input.update_torch_tensor(new_grad_input)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGeluBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_input.descriptor,
            input.descriptor,
            grad_output.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input, grad_output, grad_input]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGeluBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, grad_input.device)

    def lib_gelu_backward():
        check_error(
            LIBINFINIOP.infiniopGeluBackward(
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
        debug(
            grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol
        )
    assert torch.allclose(
        grad_input.actual_tensor(), grad_input.torch_tensor(), atol=atol, rtol=rtol
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: gelu_backward(input.torch_tensor(), grad_output.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gelu_backward(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyGeluBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        if device == InfiniDeviceEnum.ILUVATAR:
            _TENSOR_DTYPES = [
                InfiniDtype.BF16,
                InfiniDtype.F16,
                InfiniDtype.F32,
            ]
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
