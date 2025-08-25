import torch
import ctypes
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
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]


def row_major_strides(shape):
    """生成张量的行优先(C风格)stride"""
    stride = 1
    strides = [1]
    for dim in reversed(shape[1:]):
        stride *= dim
        strides.insert(0, stride)
    return strides


def column_major_strides(shape):
    """生成张量的列优先(Fortran风格)stride"""
    stride = 1
    strides = [stride]
    for dim in shape[:-1]:
        stride *= dim
        strides.append(stride)
    return strides


_TEST_CASES = [
    # (input_shape, dim, index_shape, input_stride, output_stride)
    # 基本测试用例 - 连续内存布局
    # 注意：在 torch.gather 中，输出形状等于索引形状
    ((4, 5), 0, (2, 5), None, None),  # 在第0维进行gather
    ((4, 5), 1, (4, 3), None, None),  # 在第1维进行gather
    ((3, 4, 5), 0, (2, 4, 5), None, None),  # 3D张量在第0维进行gather
    ((3, 4, 5), 1, (3, 2, 5), None, None),  # 3D张量在第1维进行gather
    ((3, 4, 5), 2, (3, 4, 3), None, None),  # 3D张量在第2维进行gather
]

# Data types used for testing - 所有合法类型
_TENSOR_DTYPES = [
    InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64, InfiniDtype.BF16,
    InfiniDtype.I8, InfiniDtype.I16, InfiniDtype.I32, InfiniDtype.I64,
    InfiniDtype.U8, InfiniDtype.U16, InfiniDtype.U32, InfiniDtype.U64,
    InfiniDtype.BOOL,
]

# Index data types
_INDEX_DTYPES = [InfiniDtype.I32, InfiniDtype.I64]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-15},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
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


def gather_torch(input_tensor, dim, index):
    return torch.gather(input_tensor, dim, index)


def test(
    handle, torch_device, input_shape, dim, index_shape, input_stride=None, output_stride=None, inplace=Inplace.OUT_OF_PLACE, dtype=InfiniDtype.F32, sync=None
):
    # 在 torch.gather 中，输出形状等于索引形状
    output_shape = index_shape
    print(
        f"Testing Gather on {InfiniDeviceNames[torch_device]} with input_shape:{input_shape} output_shape:{output_shape} dim:{dim} index_shape:{index_shape} input_stride:{input_stride} output_stride:{output_stride} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace.name}"
    )

    input_tensor = TestTensor(input_shape, input_stride, dtype, torch_device)
    
    if inplace == Inplace.INPLACE:
        if input_stride != output_stride or input_shape != output_shape:
            return
        output = input_tensor
    else:
        output = TestTensor(output_shape, output_stride, dtype, torch_device, mode="zeros")
    
    # 创建索引张量，确保索引值在有效范围内
    max_index = input_shape[dim] - 1
    index_data = torch.randint(0, max_index + 1, index_shape, dtype=torch.int32)  # 使用int32类型索引
    index = TestTensor.from_torch(index_data, InfiniDtype.I32, torch_device)
    
    # PyTorch参考实现
    output_torch = gather_torch(input_tensor.torch_tensor(), dim, index.torch_tensor().long())

    if sync is not None:
        sync()

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGatherDescriptor(
            handle, ctypes.byref(descriptor), input_tensor.descriptor, output.descriptor, dim, index.descriptor
        )
    )

    def lib_gather():
        check_error(LIBINFINIOP.infiniopGather(
            descriptor, 
            None, 0, 
            output.data(), 
            input_tensor.data(), 
            index.data(), 
            None
        ))

    lib_gather()

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    # Note: This should be done AFTER the gather operation to avoid accessing freed memory
    for tensor in [input_tensor, output, index]:
        tensor.destroy_desc()

    # 验证结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    actual_output = output.actual_tensor()
    if DEBUG:
        debug(actual_output, output_torch, atol=atol, rtol=rtol)
    assert torch.allclose(actual_output, output_torch, atol=atol, rtol=rtol)

    # 性能分析
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: gather_torch(input_tensor.torch_tensor(), dim, index.torch_tensor()), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gather(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyGatherDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    # 配置测试选项
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # 执行测试
    # Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES
    test_cases_with_inplace = [
        test_case + (inplace_item,)
        for test_case in _TEST_CASES
        for inplace_item in _INPLACE
    ]

    for device in get_test_devices(args):
        test_operator(device, test, test_cases_with_inplace, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")