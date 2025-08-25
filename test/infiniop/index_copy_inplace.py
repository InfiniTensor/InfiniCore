import torch
import ctypes
from enum import Enum, auto
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


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


_INPLACE = [Inplace.OUT_OF_PLACE, Inplace.INPLACE]


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
    # (input_shape, output_shape, dim, index_shape, input_stride, output_stride)
    # 基本测试用例 - 连续内存布局
    ((4, 5), (4, 5), 0, (2,), None, None),  # 在第0维进行索引复制
    ((4, 5), (4, 5), 1, (3,), None, None),  # 在第1维进行索引复制
    ((3, 4, 5), (3, 4, 5), 0, (2,), None, None),  # 3D张量在第0维进行索引复制
    ((3, 4, 5), (3, 4, 5), 1, (2,), None, None),  # 3D张量在第1维进行索引复制
    ((3, 4, 5), (3, 4, 5), 2, (3,), None, None),  # 3D张量在第2维进行索引复制
    
    # 任意步长测试用例 - 非连续内存布局
    ((4, 6), (4, 6), 0, (2,), (6, 1), (6, 1)),  
    ((4, 6), (4, 6), 1, (3,), (6, 1), (6, 1)),  
    ((4, 6), (4, 6), 0, (2,), (1, 4), (1, 4)),  
    ((4, 6), (4, 6), 1, (3,), (1, 4), (1, 4)),  
    
    ((3, 4, 5), (3, 4, 5), 0, (2,), row_major_strides((3, 4, 5)), row_major_strides((3, 4, 5))),
    ((3, 4, 5), (3, 4, 5), 1, (2,), row_major_strides((3, 4, 5)), column_major_strides((3, 4, 5))),
    ((3, 4, 5), (3, 4, 5), 2, (3,), column_major_strides((3, 4, 5)), row_major_strides((3, 4, 5))),

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


def index_copy_inplace_torch(output, input_tensor, dim, index):
    """PyTorch参考实现"""
    output.index_copy_(dim, index, input_tensor)


def test(
    handle, torch_device, input_shape, output_shape, dim, index_shape, input_stride=None, output_stride=None, inplace=Inplace.OUT_OF_PLACE, dtype=InfiniDtype.F32, sync=None
):
    print(
        f"Testing IndexCopyInplace on {InfiniDeviceNames[torch_device]} with input_shape:{input_shape} output_shape:{output_shape} dim:{dim} index_shape:{index_shape} input_stride:{input_stride} output_stride:{output_stride} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace.name}"
    )

    input_tensor = TestTensor(input_shape, input_stride, dtype, torch_device)
    
    # 根据inplace参数创建output张量
    if inplace == Inplace.INPLACE:
        if input_stride != output_stride:
            return
        output = input_tensor
    else:
        output = TestTensor(output_shape, output_stride, dtype, torch_device, mode="ones")
    
    # 创建索引张量，确保索引值在有效范围内
    max_index = output_shape[dim] - 1
    index_data = torch.randint(0, max_index + 1, index_shape, dtype=torch.int64)  # PyTorch需要long类型索引
    index = TestTensor.from_torch(index_data, InfiniDtype.I64, torch_device)
    
    # 调整输入张量的形状以匹配索引复制的要求
    # 输入张量在指定维度上的大小应该等于索引的大小
    adjusted_input_shape = list(input_shape)
    adjusted_input_shape[dim] = index_shape[0]  # 索引的大小
    
    # 重新创建输入张量
    adjusted_input_stride = None
    if input_stride is not None:
        adjusted_input_stride = list(input_stride)
        # 步长不需要调整，因为它们对应于张量的维度结构
    input_tensor = TestTensor(adjusted_input_shape, adjusted_input_stride, dtype, torch_device)
    
    # 保存输出张量的副本用于PyTorch参考计算
    output_torch = output.torch_tensor().clone()
    
    # PyTorch参考实现
    index_copy_inplace_torch(output_torch, input_tensor.torch_tensor(), dim, index.torch_tensor())

    if sync is not None:
        sync()

    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateIndexCopyInplaceDescriptor(
            handle, ctypes.byref(descriptor), input_tensor.descriptor, output.descriptor, dim, index.descriptor
        )
    )


    def lib_index_copy_inplace():
        check_error(LIBINFINIOP.infiniopIndexCopyInplace(descriptor, None, 0, input_tensor.data(), output.data(), index.data(), None))

    lib_index_copy_inplace()

    # 验证结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output_torch, atol=atol, rtol=rtol)
    assert torch.allclose(output.actual_tensor(), output_torch, atol=atol, rtol=rtol)

    # 性能分析
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: index_copy_inplace_torch(output.torch_tensor(), input_tensor.torch_tensor(), dim, index.torch_tensor()), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_index_copy_inplace(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyIndexCopyInplaceDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    # 配置测试选项
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES
    test_cases_with_inplace = [
        test_case + (inplace_item,)
        for test_case in _TEST_CASES
        for inplace_item in _INPLACE
    ]

    for device in get_test_devices(args):
        test_operator(device, test, test_cases_with_inplace, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")