# import ctypes
# from ctypes import c_uint64, c_float
# from enum import Enum, auto

# import torch
# from libinfiniop import (
#     LIBINFINIOP,
#     InfiniDeviceNames,
#     InfiniDtype,
#     InfiniDtypeNames,
#     TestTensor,
#     TestWorkspace,
#     check_error,
#     debug,
#     get_args,
#     get_test_devices,
#     get_tolerance,
#     infiniopOperatorDescriptor_t,
#     profile_operation,
#     test_operator,
# )

# # ==============================================================================
# #  Configuration (Internal Use Only)
# # ==============================================================================
# # These are not meant to be imported from other modules
# _TEST_CASES_ = [
#     # tensor_shape, negative_slope
#     ((1, 3), 0.01),
#     ((3, 3), 0.1),
#     ((32, 20, 512), 0.2),
#     ((33, 333, 333), 0.01),
#     ((32, 256, 112, 112), 0.1),
#     ((3, 3, 13, 9, 17), 0.2),
# ]


# class Inplace(Enum):
#     OUT_OF_PLACE = auto()
#     INPLACE_X = auto()


# # Inplace options applied for each test case in _TEST_CASES_
# _INPLACE = [
#     Inplace.OUT_OF_PLACE,
#     Inplace.INPLACE_X,
# ]

# # Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
# _TEST_CASES = [
#     test_case + (inplace_item,)
#     for test_case in _TEST_CASES_
#     for inplace_item in _INPLACE
# ]

# # Data types used for testing
# _TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# # Tolerance map for different data types
# _TOLERANCE_MAP = {
#     InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
#     InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
#     InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
# }

# DEBUG = False
# PROFILE = False
# NUM_PRERUN = 10
# NUM_ITERATIONS = 1000


# def leaky_relu(x, negative_slope):
#     return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope).to(x.dtype)


# def test(
#     handle, device, shape, negative_slope, inplace=Inplace.OUT_OF_PLACE, dtype=torch.float16, sync=None
# ):
#     x_torch_tensor = torch.rand(shape) * 2 - 1

#     x = TestTensor(
#         shape,
#         x_torch_tensor.stride(),
#         dtype,
#         device,
#         mode="manual",
#         set_tensor=x_torch_tensor,
#     )

#     if inplace == Inplace.INPLACE_X:
#         y = x
#     else:
#         y = TestTensor(shape, None, dtype, device)

#     if y.is_broadcast():
#         return

#     print(
#         f"Testing LeakyReLU on {InfiniDeviceNames[device]} with shape:{shape} dtype:{InfiniDtypeNames[dtype]} negative_slope:{negative_slope} inplace: {inplace}"
#     )

#     ans = leaky_relu(x.torch_tensor(), negative_slope)

#     if sync is not None:
#         sync()

#     descriptor = infiniopOperatorDescriptor_t()
#     check_error(
#         LIBINFINIOP.infiniopCreateLeakyReLUDescriptor(
#             handle, ctypes.byref(descriptor), y.descriptor, x.descriptor, c_float(negative_slope)
#         )
#     )

#     # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
#     for tensor in [x, y]:
#         tensor.destroy_desc()

#     workspace_size = c_uint64(0)
#     check_error(
#         LIBINFINIOP.infiniopGetLeakyReLUWorkspaceSize(
#             descriptor, ctypes.byref(workspace_size)
#         )
#     )
#     workspace = TestWorkspace(workspace_size.value, y.device)

#     def lib_leaky_relu():
#         LIBINFINIOP.infiniopLeakyReLU(
#             descriptor, workspace.data(), workspace.size(), y.data(), x.data(), None
#         )

#     lib_leaky_relu()

#     atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
#     if DEBUG:
#         debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
#     assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

#     # Profiling workflow
#     if PROFILE:
#         # fmt: off
#         profile_operation("PyTorch", lambda: leaky_relu(x.torch_tensor(), negative_slope), device, NUM_PRERUN, NUM_ITERATIONS)
#         profile_operation("    lib", lambda: lib_leaky_relu(), device, NUM_PRERUN, NUM_ITERATIONS)
#         # fmt: on

#     check_error(LIBINFINIOP.infiniopDestroyLeakyReLUDescriptor(descriptor))


# if __name__ == "__main__":
#     args = get_args()

#     # Configure testing options
#     DEBUG = args.debug
#     PROFILE = args.profile
#     NUM_PRERUN = args.num_prerun
#     NUM_ITERATIONS = args.num_iterations

#     for device in get_test_devices(args):
#         test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

#     print("\033[92mTest passed!\033[0m")


import ctypes
from ctypes import c_uint64, c_float
from enum import Enum, auto

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
    # tensor_shape, negative_slope, inplace
    # TODO: Uncomment the following line.
    # ((), 0.01),
    ((1, 3), 0.01),
    ((3, 3), 0.1),
    ((32, 20, 512), 0.2),
    ((33, 333, 333), 0.01),
    ((32, 256, 112, 112), 0.1),
    ((3, 3, 13, 9, 17), 0.02),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_INPUT = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_INPUT,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def leaky_relu(x, negative_slope=0.01):
    return torch.nn.functional.leaky_relu(x, negative_slope=negative_slope).to(x.dtype)


def test(
    handle,
    device,
    shape,
    negative_slope,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    # Generate test data with both positive and negative values to test LeakyReLU properly
    input_torch_tensor = torch.randn(shape) * 2  # Range around [-2, 2]

    input_tensor = TestTensor(
        shape,
        input_torch_tensor.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=input_torch_tensor,
    )

    if inplace == Inplace.INPLACE_INPUT:
        output = input_tensor
    else:
        output = TestTensor(shape, None, dtype, device)

    if output.is_broadcast():
        return

    print(
        f"Testing LeakyReLU on {InfiniDeviceNames[device]} with shape:{shape} dtype:{InfiniDtypeNames[dtype]} negative_slope:{negative_slope} inplace: {inplace}"
    )

    ans = leaky_relu(input_tensor.torch_tensor(), negative_slope)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLeakyReLUDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input_tensor.descriptor,
            c_float(negative_slope),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, output]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLeakyReLUWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_leaky_relu():
        LIBINFINIOP.infiniopLeakyReLU(
            descriptor,
            workspace.data(),
            workspace.size(),
            output.data(),
            input_tensor.data(),
            None,
        )

    lib_leaky_relu()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(output.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: leaky_relu(input_tensor.torch_tensor(), negative_slope), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_leaky_relu(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyLeakyReLUDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mLeakyReLU Test passed!\033[0m")
