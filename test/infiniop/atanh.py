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
    # shape, a_stride, y_stride
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
]

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()

_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_A,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# atanh typically supports floating point types
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-6, "rtol": 1e-6},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def atanh_torch(y, a):
    torch.atanh(a, out=y)

def test(
    handle,
    device,
    shape,
    a_stride=None,
    y_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    # Initialize input tensor
    a = TestTensor(shape, a_stride, dtype, device)
    
    # Crucial: clamp values to (-1, 1) to avoid NaN/Inf for atanh
    with torch.no_grad():
        a.torch_tensor().clamp_(-0.99, 0.99)
        # Keep underlying data in sync for all devices (including CPU)
        a.actual_tensor().copy_(a.torch_tensor())

    if inplace == Inplace.INPLACE_A:
        if a_stride != y_stride:
            return
        y = a
    else:
        y = TestTensor(shape, y_stride, dtype, device, mode="ones")

    if y.is_broadcast():
        return

    print(
        f"Testing Atanh on {InfiniDeviceNames[device]} with shape:{shape} a_stride:{a_stride} y_stride:{y_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # Reference calculation
    atanh_torch(y.torch_tensor(), a.torch_tensor())

    if sync is not None:
        sync()

    # Create descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAtanhDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            a.descriptor,
        )
    )

    # Invalidate descriptors to ensure kernel uses its own internal state
    for tensor in [a, y]:
        tensor.destroy_desc()

    # Workspace management
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAtanhWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_atanh():
        check_error(
            LIBINFINIOP.infiniopAtanh(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                a.data(),
                None,
            )
        )

    # Run library function
    lib_atanh()

    # Validate results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling
    if PROFILE:
        profile_operation("PyTorch", lambda: atanh_torch(y.torch_tensor(), a.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("   lib", lambda: lib_atanh(), device, NUM_PRERUN, NUM_ITERATIONS)
        
    check_error(LIBINFINIOP.infiniopDestroyAtanhDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mAtanh Test passed!\033[0m")