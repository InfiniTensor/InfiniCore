import torch
import ctypes
from ctypes import c_uint64
from ctypes import c_uint8
from ctypes import c_uint16
from ctypes import c_uint32
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
    # shape, a_stride, b_stride, c_stride, condition_stride
    ((13, 4), None, None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None, None),
    ((13, 4, 4), None, None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None, None),
    ((16, 5632), None, None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None, None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
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

# Data types used for testing - support all legal types
_TENSOR_DTYPES = [
    InfiniDtype.F16, 
    InfiniDtype.F32, 
    InfiniDtype.F64, 
    InfiniDtype.BF16,
    InfiniDtype.BOOL,
    InfiniDtype.I8,
    InfiniDtype.I16,
    InfiniDtype.I32,
    InfiniDtype.I64,
    # InfiniDtype.U8,
    # InfiniDtype.U16,
    # InfiniDtype.U32,
    # InfiniDtype.U64
    # InfiniDtype.F8
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.BOOL: {"atol": 0, "rtol": 0},
    InfiniDtype.I8: {"atol": 0, "rtol": 0},
    InfiniDtype.I16: {"atol": 0, "rtol": 0},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
    # InfiniDtype.U8: {"atol": 0, "rtol": 0},
    # InfiniDtype.U16: {"atol": 0, "rtol": 0},
    # InfiniDtype.U32: {"atol": 0, "rtol": 0},
    # InfiniDtype.U64: {"atol": 0, "rtol": 0}
    # InfiniDtype.F8: {"atol": 1e-3, "rtol": 1e-3}
}

DEBUG = True
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def where_op(c, a, b, condition):
    """PyTorch reference implementation of where operation"""
    torch.where(condition, a, b, out=c)
# def where_op(c, a, b, condition):
#     """PyTorch reference implementation of where operation"""
#     # 检查数据类型并进行必要的转换
#     unsupported_types = (torch.uint16, torch.uint32, torch.uint64)
    
#     if a.dtype in unsupported_types or b.dtype in unsupported_types or c.dtype in unsupported_types:
#         # 将不支持的类型转换为对应的兼容类型
#         def get_compatible_dtype(dtype):
#             if dtype == torch.uint16:
#                 return torch.int16
#             elif dtype == torch.uint32:
#                 return torch.int32
#             elif dtype == torch.uint64:
#                 return torch.int64
#             else:
#                 return dtype
        
#         a_converted = a.to(get_compatible_dtype(a.dtype)) if a.dtype in unsupported_types else a
#         b_converted = b.to(get_compatible_dtype(b.dtype)) if b.dtype in unsupported_types else b
#         c_converted = c.to(get_compatible_dtype(c.dtype)) if c.dtype in unsupported_types else c
        
#         # 调用torch.where
#         torch.where(condition, a_converted, b_converted, out=c_converted)
        
#         # 如果需要，将结果转换回原始类型
#         if c.dtype in unsupported_types:
#             c.copy_(c_converted.to(c.dtype))
#     else:
#         # 对于其他支持的类型，直接调用torch.where
#         torch.where(condition, a, b, out=c)

def test(
    handle,
    device,
    shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    condition_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float32,
    sync=None,
):
    # Create tensors with specified data type
    a = TestTensor(shape, a_stride, dtype, device)
    b = TestTensor(shape, b_stride, dtype, device)
    
    # Create condition tensor (always bool type)
    condition = TestTensor(shape, condition_stride, InfiniDtype.BOOL, device)
    
    if inplace == Inplace.INPLACE_A:
        if a_stride != c_stride:
            return
        c = a
    elif inplace == Inplace.INPLACE_B:
        if c_stride != b_stride:
            return
        c = b
    else:
        c = TestTensor(shape, c_stride, dtype, device, mode="ones")

    if c.is_broadcast():
        return

    print(
        f"Testing Where on {InfiniDeviceNames[device]} with shape:{shape} "
        f"a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
        f"condition_stride:{condition_stride} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # Execute PyTorch reference implementation
    where_op(c.torch_tensor(), a.torch_tensor(), b.torch_tensor(), condition.torch_tensor())

    if sync is not None:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
            condition.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [a, b, c, condition]:
        tensor.destroy_desc()

    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetWhereWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)



    def libwhere():
        check_error(
            LIBINFINIOP.infiniopWhere(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                a.data(),
                b.data(),
                condition.data(),
                None,
            )
        )

    libwhere()

    # Sync the torch_tensor with actual_tensor after Infiniop operation
    # Copy data from actual_tensor to torch_tensor to ensure consistency
    c.torch_tensor().copy_(c.actual_tensor())

    # Verify results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: where_op(c.torch_tensor(), a.torch_tensor(), b.torch_tensor(), condition.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: libwhere(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    check_error(LIBINFINIOP.infiniopDestroyWhereDescriptor(descriptor))


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
