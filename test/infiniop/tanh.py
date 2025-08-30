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
    get_sync_func,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ========================================================================
#  Configuration (Internal Use Only)
# ========================================================================
_TEST_CASES_ = [
    # shape, input_stride, output_stride
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), None),
    ((16, 5632), None, None),
    ((16, 5632), (10240, 1), (10240, 1)),
    ((4, 4, 5632), None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
]

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_INPUT = auto()

_INPLACE = [
    Inplace.OUT_OF_PLACE, 
    Inplace.INPLACE_INPUT,
]

_TEST_CASES = [
    test_case + (inplace,)
    for test_case in _TEST_CASES_
    for inplace in _INPLACE
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG       = False
PROFILE     = False
NUM_PRERUN  = 10
NUM_ITERATIONS = 1000


def tanh(output, input):
    output.copy_(torch.tanh(input))

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
    input = TestTensor(shape, input_stride, dtype, device)
    if inplace == Inplace.INPLACE_INPUT:
        if input_stride != output_stride:
            return
        output = input
    else:
        output = TestTensor(shape, output_stride, dtype, device, mode="ones")

    if output.is_broadcast():
        return

    print(
        f"Testing Tanh on {InfiniDeviceNames[device]} with shape:{shape} input_stride:{input_stride} output_stride:{output_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    tanh(output.torch_tensor(), input.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTanhDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input, output]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTanhWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_tanh():
        check_error(
            LIBINFINIOP.infiniopTanh(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output.data(),
                input.data(),
                None,
            )
        )

    lib_tanh()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
        
    assert torch.allclose(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: tanh(output.torch_tensor(), input.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_tanh(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyTanhDescriptor(descriptor))


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
