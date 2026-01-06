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

# ==============================================================================
# Configuration
# ==============================================================================
_TEST_CASES_ = [
    # shape, input_stride, output_stride
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((16, 5632), None, None),
    ((16, 5632), (13312, 1), (13312, 1)),
    ((13, 16, 2), (128, 4, 1), (64, 4, 1)),
    ((4, 4, 5632), None, None),
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

# Reciprocal usually outputs floats; Integer types are often not supported or special-cased
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def reciprocal(y, x):
    torch.reciprocal(x, out=y)

def test(
    handle,
    device,
    shape,
    in_stride=None,
    out_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    # Initialize input 'x'
    # Use 'random' mode but ensure values are not near zero to avoid infinity
    x = TestTensor(shape, in_stride, dtype, device)
    
    if inplace == Inplace.INPLACE:
        if in_stride != out_stride:
            return
        y = x
    else:
        y = TestTensor(shape, out_stride, dtype, device)

    if y.is_broadcast():
        return

    print(
        f"Testing Reciprocal on {InfiniDeviceNames[device]} with shape:{shape} "
        f"in_stride:{in_stride} out_stride:{out_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # Calculate ground truth using PyTorch
    reciprocal(y.torch_tensor(), x.torch_tensor())

    if sync is not None:
        sync()

    # Create Descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReciprocalDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
        )
    )

    # Invalidate descriptors as per framework requirement
    for tensor in [x, y]:
        tensor.destroy_desc()

    # Workspace allocation
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReciprocalWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_reciprocal():
        check_error(
            LIBINFINIOP.infiniopReciprocal(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                x.data(),
                None,
            )
        )

    lib_reciprocal()

    # Verification
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling
    if PROFILE:
        profile_operation("PyTorch", lambda: reciprocal(y.torch_tensor(), x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("   lib", lambda: lib_reciprocal(), device, NUM_PRERUN, NUM_ITERATIONS)
        
    check_error(LIBINFINIOP.infiniopDestroyReciprocalDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")