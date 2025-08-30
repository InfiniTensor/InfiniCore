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
    # shape, condition_stride, a_stride, b_stride, c_stride
    # 基本形状测试
    ((10,), None, None, None, None),
    ((5, 10), None, None, None, None),
    ((2, 3, 4), None, None, None, None),
    ((4, 5, 6), None, None, None, None),
    # 不同步长测试
    ((10, 10), (10, 1), None, None, None),
    ((10, 10), None, (10, 1), None, None),
    ((10, 10), None, None, (10, 1), None),
    ((10, 10), None, None, None, (10, 1)),
    # 奇怪形状测试
    ((7, 13), None, None, None, None),  # 质数维度
    ((3, 5, 7), None, None, None, None),  # 三维质数
    ((11, 17, 23), None, None, None, None),  # 更大质数
    # 非标准形状测试
    ((1, 1), None, None, None, None),  # 最小形状
    ((1, 100), None, None, None, None),  # 单行
    ((100, 1), None, None, None, None),  # 单列
    ((64, 64), None, None, None, None),  # 2的幂次
    ((16, 16, 16), None, None, None, None),  # 三维2的幂次
    # 大形状测试
    ((100, 100), None, None, None, None),
    ((32, 32, 32), None, None, None, None),
    # 广播测试 - 这些会被跳过，但保留作为潜在的扩展
    ((10,), (0,), None, None, None),  # 广播condition
    ((5, 10), None, (0, 1), None, None),  # 广播a
    ((5, 10), None, None, (0, 1), None),  # 广播b
]


# 暂时只测试浮点类型，确认逻辑正确后再扩展到整数类型
_TENSOR_DTYPES = [
    InfiniDtype.F16,
    InfiniDtype.F32,
    InfiniDtype.F64,
    InfiniDtype.BF16,
    InfiniDtype.I8,
    InfiniDtype.I16,
    InfiniDtype.I32,
    InfiniDtype.I64,
    # InfiniDtype.U8, InfiniDtype.U16, InfiniDtype.U32, InfiniDtype.U64,
    InfiniDtype.BOOL,
]


_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-6},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-14},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
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


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


_INPLACE = [
    Inplace.INPLACE_A,
    Inplace.INPLACE_B,
    Inplace.OUT_OF_PLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def where(c, condition, a, b):
    """Where operation: c[i] = condition[i] ? a[i] : b[i]"""
    result = torch.where(condition.to(torch.bool), a, b)
    c.copy_(result)


def test(
    handle,
    device,
    shape,
    condition_stride=None,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F32,
    sync=None,
):
    # Create input tensors a and b with specified dtype
    # For unsigned integer types, we need to be careful about random generation
    if dtype in [InfiniDtype.U8, InfiniDtype.U16, InfiniDtype.U32, InfiniDtype.U64]:
        # Use a smaller range for unsigned types to avoid overflow
        a = TestTensor(shape, a_stride, dtype, device, mode="random", scale=10, bias=0)
        b = TestTensor(shape, b_stride, dtype, device, mode="random", scale=10, bias=0)
        condition = TestTensor(
            shape, condition_stride, dtype, device, mode="random", scale=10, bias=0
        )
    elif dtype in [InfiniDtype.I8, InfiniDtype.I16, InfiniDtype.I32, InfiniDtype.I64]:
        # Use a reasonable range for signed integer types
        a = TestTensor(
            shape, a_stride, dtype, device, mode="random", scale=100, bias=-50
        )
        b = TestTensor(
            shape, b_stride, dtype, device, mode="random", scale=100, bias=-50
        )
        condition = TestTensor(
            shape, condition_stride, dtype, device, mode="random", scale=100, bias=-50
        )
    else:
        # For floating point and bool types, use default random generation
        a = TestTensor(shape, a_stride, dtype, device)
        b = TestTensor(shape, b_stride, dtype, device)
        condition = TestTensor(shape, condition_stride, dtype, device)
    # Handle inplace operations
    if inplace == Inplace.INPLACE_A:
        if a_stride != c_stride:
            return
        c = a
    elif inplace == Inplace.INPLACE_B:
        if b_stride != c_stride:
            return
        c = b
    else:
        c = TestTensor(shape, c_stride, dtype, device)

    # Skip broadcast cases for now
    if (
        c.is_broadcast()
        or condition.is_broadcast()
        or a.is_broadcast()
        or b.is_broadcast()
    ):
        return

    print(
        f"Testing Where on {InfiniDeviceNames[device]} with shape:{shape} "
        f"condition_stride:{condition_stride} a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # Compute reference result using PyTorch
    where(
        c.torch_tensor(), condition.torch_tensor(), a.torch_tensor(), b.torch_tensor()
    )

    if sync is not None:
        sync()

    # Store expected result before library operation
    expected_result = c.torch_tensor().clone()

    # Create descriptor
    descriptor = infiniopOperatorDescriptor_t()
    print(
        a.torch_tensor().dtype,
        b.torch_tensor().dtype,
        condition.torch_tensor().dtype,
        c.torch_tensor().dtype,
    )
    check_error(
        LIBINFINIOP.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            condition.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetWhereWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_where():
        check_error(
            LIBINFINIOP.infiniopWhere(
                descriptor,
                workspace.data() if workspace is not None else None,
                workspace_size.value,
                c.data(),
                condition.data(),
                a.data(),
                b.data(),
                None,
            )
        )

    # Execute library operation
    lib_where()

    # Destroy the tensor descriptors
    for tensor in [condition, a, b, c]:
        tensor.destroy_desc()

    # Check results with better error reporting
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    # # Always print debug info for failed cases
    # print(f"Condition values: {condition.torch_tensor().flatten()[:10]}")
    # print(f"A values: {a.torch_tensor().flatten()[:10]}")
    # print(f"B values: {b.torch_tensor().flatten()[:10]}")
    # print(f"Expected result: {expected_result.flatten()[:10]}")
    # print(f"Actual result: {c.actual_tensor().flatten()[:10]}")

    # if DEBUG:
    #     print(f"Expected result shape: {expected_result.shape}")
    #     print(f"Actual result shape: {c.actual_tensor().shape}")
    #     print(f"Expected result dtype: {expected_result.dtype}")
    #     print(f"Actual result dtype: {c.actual_tensor().dtype}")
    #     debug(c.actual_tensor(), expected_result, atol=atol, rtol=rtol)

    # Use torch.equal for exact comparison for integer and boolean types
    if dtype in [
        InfiniDtype.I8,
        InfiniDtype.I16,
        InfiniDtype.I32,
        InfiniDtype.I64,
        InfiniDtype.U8,
        InfiniDtype.U16,
        InfiniDtype.U32,
        InfiniDtype.U64,
        InfiniDtype.BOOL,
    ]:
        if not torch.equal(c.actual_tensor(), expected_result):
            print(f"Exact comparison failed for {InfiniDtypeNames[dtype]}")
            print(
                f"Max absolute difference: {torch.max(torch.abs(c.actual_tensor() - expected_result))}"
            )
            assert False, f"Results don't match exactly for {InfiniDtypeNames[dtype]}"
    else:
        if not torch.allclose(c.actual_tensor(), expected_result, atol=atol, rtol=rtol):
            print(f"Tolerance comparison failed for {InfiniDtypeNames[dtype]}")
            print(
                f"Max absolute difference: {torch.max(torch.abs(c.actual_tensor() - expected_result))}"
            )
            print(f"Tolerance: atol={atol}, rtol={rtol}")
            assert (
                False
            ), f"Results don't match within tolerance for {InfiniDtypeNames[dtype]}"

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: where(c.torch_tensor(), condition.torch_tensor(), a.torch_tensor(), b.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_where(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    # Clean up
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

    print("\033[92mWhere test passed!\033[0m")
