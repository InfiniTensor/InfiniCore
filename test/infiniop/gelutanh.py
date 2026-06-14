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

_TEST_CASES_ = [
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None),
    ((16, 5632), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64]

_TOLERANCE_MAP = {
    InfiniDtype.BF16: {"atol": 2e-2, "rtol": 2e-2},
    InfiniDtype.F16: {"atol": 2e-3, "rtol": 2e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F64: {"atol": 1e-6, "rtol": 1e-6},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def test(
    handle,
    device,
    shape,
    input_stride=None,
    output_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    if device == InfiniDeviceEnum.ILUVATAR and (
        input_stride is not None or output_stride is not None
    ):
        return
    if device == InfiniDeviceEnum.MOORE and dtype == InfiniDtype.F64:
        return

    input = TestTensor(shape, input_stride, dtype, device)
    if inplace == Inplace.INPLACE:
        if input_stride != output_stride:
            return
        output = input
    else:
        output = TestTensor(shape, output_stride, dtype, device, mode="ones")

    if output.is_broadcast():
        return

    print(
        f"Testing GeluTanh on {InfiniDeviceNames[device]} with shape:{shape} "
        f"input_stride:{input_stride} output_stride:{output_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    new_tensor = torch.nn.functional.gelu(input.torch_tensor(), approximate="tanh")
    output.update_torch_tensor(new_tensor)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGeluTanhDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input.descriptor,
        )
    )

    for tensor in [input, output]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGeluTanhWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_gelutanh():
        check_error(
            LIBINFINIOP.infiniopGeluTanh(
                descriptor,
                workspace.data(),
                workspace.size(),
                output.data(),
                input.data(),
                None,
            )
        )

    lib_gelutanh()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(
        output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol
    )

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch.nn.functional.gelu(input.torch_tensor(), approximate="tanh"),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_gelutanh(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyGeluTanhDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
