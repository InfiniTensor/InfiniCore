import ctypes
from ctypes import c_uint64
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

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# HardSwish 是分段线性（含除法），数值较稳定；容差与 GeLU 保持一致或更严格均可
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

_hswish = torch.nn.Hardswish(inplace=False)


def hardswish(x):
    """
    Reference HardSwish using PyTorch:
      hswish(x) = x * clamp(x + 3, 0, 6) / 6
    """
    # return torch.nn.functional.hardswish(x).to(x.dtype)
    return _hswish(x).to(x.dtype)


def test(
    handle, device, shape, inplace=Inplace.OUT_OF_PLACE, dtype=torch.float16, sync=None
):
    # Sample inputs from [-2, 2]，覆盖线性区和中间平滑区；可根据需要扩大范围
    input_torch_tensor = torch.rand(shape) * 4 - 2

    input_tensor = TestTensor(
        shape,
        input_torch_tensor.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=input_torch_tensor,
    )

    if inplace == Inplace.INPLACE_INPUT:
        output_tensor = input_tensor
    else:
        output_tensor = TestTensor(shape, None, dtype, device)

    if output_tensor.is_broadcast():
        return

    print(
        f"Testing HardSwish on {InfiniDeviceNames[device]} with shape:{shape} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # PyTorch reference
    ans = hardswish(input_tensor.torch_tensor())

    if sync is not None:
        sync()

    # Create HardSwish descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHardSwishDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
        )
    )

    # Invalidate desc shapes/strides to ensure kernel uses runtime args
    for tensor in [input_tensor, output_tensor]:
        tensor.destroy_desc()

    # Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHardSwishWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output_tensor.device)

    def lib_hardswish():
        LIBINFINIOP.infiniopHardSwish(
            descriptor,
            workspace.data(),
            workspace.size(),
            output_tensor.data(),
            input_tensor.data(),
            None,
        )

    # Run lib op
    lib_hardswish()

    # Verify
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output_tensor.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(output_tensor.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling (optional)
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: hardswish(input_tensor.torch_tensor()),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_hardswish(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    # Clean up
    check_error(LIBINFINIOP.infiniopDestroyHardSwishDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mHardSwish test passed!\033[0m")
