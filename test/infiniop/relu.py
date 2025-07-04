import ctypes
from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
from enum import Enum, auto

import torch
from libinfiniop import (
    check_error,
    create_workspace,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    profile_operation,
    test_operator,
    to_tensor,
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


class ReluDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopReluDescriptor_t = POINTER(ReluDescriptor)


def relu(x):
    return torch.nn.functional.relu(x).to(x.dtype)


def test(
    lib,
    handle,
    torch_device,
    tensor_shape,
    inplace=Inplace.OUT_OF_PLACE,
    tensor_dtype=torch.float16,
    sync=None,
):
    print(
        f"Testing Relu on {torch_device} with tensor_shape:{tensor_shape} dtype:{tensor_dtype} inplace: {inplace}"
    )

    x = torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device) * 2 - 1
    y = (
        torch.rand(tensor_shape, dtype=tensor_dtype).to(torch_device)
        if inplace == Inplace.OUT_OF_PLACE
        else x
    )

    ans = relu(x)

    x_tensor = to_tensor(x, lib)
    y_tensor = to_tensor(y, lib) if inplace == Inplace.OUT_OF_PLACE else x_tensor

    if sync is not None:
        sync()

    descriptor = infiniopReluDescriptor_t()
    check_error(
        lib.infiniopCreateReluDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x_tensor, y_tensor]:
        tensor.destroyDesc(lib)

    workspace_size = c_uint64(0)
    check_error(
        lib.infiniopGetReluWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, y.device)

    def lib_relu():
        lib.infiniopRelu(
            descriptor,
            workspace.data_ptr() if workspace is not None else None,
            workspace_size.value,
            y_tensor.data,
            x_tensor.data,
            None,
        )

    lib_relu()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(y, ans, atol=atol, rtol=rtol)
    assert torch.allclose(y, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: relu(x), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_relu(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(lib.infiniopDestroyReluDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateReluDescriptor.restype = c_int32
    lib.infiniopCreateReluDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopReluDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
    ]
    lib.infiniopRelu.restype = c_int32
    lib.infiniopRelu.argtypes = [
        infiniopReluDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyReluDescriptor.restype = c_int32
    lib.infiniopDestroyReluDescriptor.argtypes = [
        infiniopReluDescriptor_t,
    ]

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
