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
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), None),
    ((4, 4, 5632), None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), None),
    # Test cases with different values to ensure false results
    ((2, 3), None, None, None),
    ((5,), None, None, None),
    # Scalar test cases
    ((), None, None, None),
    ((1,), None, None, None),
]

# Equal算子不支持inplace操作，因为输出是标量bool
class Inplace(Enum):
    OUT_OF_PLACE = auto()

_INPLACE = [Inplace.OUT_OF_PLACE]

# Combine test cases with inplace options
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Equal算子支持所有数据类型
_TENSOR_DTYPES = [
    InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64, InfiniDtype.BF16,
    InfiniDtype.I8, InfiniDtype.I16, InfiniDtype.I32, InfiniDtype.I64,
    InfiniDtype.U8, InfiniDtype.U16, InfiniDtype.U32, InfiniDtype.U64,
    InfiniDtype.BOOL
]

# Equal算子输出是bool，精度要求严格
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.F64: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
    InfiniDtype.I8: {"atol": 0, "rtol": 0},
    InfiniDtype.I16: {"atol": 0, "rtol": 0},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
    InfiniDtype.U8: {"atol": 0, "rtol": 0},
    InfiniDtype.U16: {"atol": 0, "rtol": 0},
    InfiniDtype.U32: {"atol": 0, "rtol": 0},
    InfiniDtype.U64: {"atol": 0, "rtol": 0},
    InfiniDtype.BOOL: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def equal(c, a, b):
    """Reference implementation using torch.equal"""
    result = torch.equal(a, b)
    # c is a scalar tensor, set its value
    c.fill_(result)


def test(
    handle,
    device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F32,
    sync=None,
):
    # Create test tensors
    a = TestTensor(shape, a_stride, dtype, device)
    
    # Create second tensor for comparison
    b = TestTensor(shape, b_stride, dtype, device)
    
    # Output is always a scalar bool tensor
    c = TestTensor((), None, InfiniDtype.BOOL, device)

    print(
        f"Testing Equal on {InfiniDeviceNames[device]} with shape:{shape} a_stride:{a_stride} b_stride:{b_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )
    
    # Get expected result using torch.equal
    expected_result = torch.equal(a.torch_tensor(), b.torch_tensor())
    ans = torch.tensor(expected_result, dtype=torch.bool, device=c.actual_tensor().device)
    


    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateEqualDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a, b, c]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetEqualWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    if PROFILE:
        profile_operation(
            lambda: LIBINFINIOP.infiniopEqual(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                b.data(),
                None,
            ),
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
    else:
        check_error(
            LIBINFINIOP.infiniopEqual(
                descriptor,
                workspace.data(),
                workspace_size.value,
                c.data(),
                a.data(),
                b.data(),
                None,
            )
        )

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, InfiniDtype.BOOL)
    if DEBUG:
        debug(c.actual_tensor(), ans, atol=atol, rtol=rtol)
    torch.testing.assert_close(
        c.actual_tensor(), ans, atol=atol, rtol=rtol
    )

    check_error(LIBINFINIOP.infiniopDestroyEqualDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Override global variables with command line arguments
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")