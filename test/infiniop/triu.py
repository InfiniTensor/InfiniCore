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
    torch_device_map,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]


_TEST_CASES = [
    # (input_shape, diagonal, input_stride, output_stride)
    # 基本测试用例 - 2D连续张量
    ((3, 3), 0, None, None),      # 3x3矩阵，默认对角线
    ((4, 4), 0, None, None),      # 4x4矩阵，默认对角线
    ((5, 3), 0, None, None),      # 5x3矩形矩阵
    ((3, 5), 0, None, None),      # 3x5矩形矩阵
    ((4, 4), 1, None, None),      # 上对角线偏移1
    ((4, 4), -1, None, None),     # 下对角线偏移1
    ((4, 4), 2, None, None),      # 上对角线偏移2
    ((4, 4), -2, None, None),     # 下对角线偏移2
    ((6, 6), 0, None, None),      # 较大矩阵
    ((2, 2), 0, None, None),      # 最小2x2矩阵
    ((1, 1), 0, None, None),      # 1x1矩阵
]

# Data types used for testing - 所有合法类型
_TENSOR_DTYPES = [
    InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.F64, InfiniDtype.BF16,
    InfiniDtype.I8, InfiniDtype.I16, InfiniDtype.I32, InfiniDtype.I64,
    InfiniDtype.U8, InfiniDtype.U16, InfiniDtype.U32, InfiniDtype.U64,
    InfiniDtype.BOOL,
]

# Tolerance mapping for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F64: {"atol": 1e-12, "rtol": 1e-12},
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


def triu_torch(input_tensor, diagonal=0):
    """PyTorch triu操作的参考实现"""
    return torch.triu(input_tensor, diagonal=diagonal)


def test(handle, device, input_shape, diagonal, input_stride, output_stride, inplace=Inplace.OUT_OF_PLACE, dtype=InfiniDtype.F32, sync=None):
    torch_device = torch.device(torch_device_map[device])
    output_shape = input_shape  # triu输出形状与输入相同
    
    print(
        f"Testing Triu on {InfiniDeviceNames[device]} with input_shape:{input_shape} diagonal:{diagonal} input_stride:{input_stride} output_stride:{output_stride} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )
    
    # 创建输入张量
    input_tensor = TestTensor(input_shape, input_stride, dtype, device)
    
    # 根据inplace参数创建输出张量
    if inplace == Inplace.INPLACE:
        if input_stride != output_stride:
            return
        output = input_tensor
    else:
        output = TestTensor(output_shape, output_stride, dtype, device, mode="zeros")
    
    # PyTorch参考实现
    output_torch = triu_torch(input_tensor.torch_tensor(), diagonal)
    
    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateTriuDescriptor(
        handle,
        ctypes.byref(descriptor),
        input_tensor.descriptor,
        output.descriptor,
        diagonal
    ))
    
    def lib_triu():
        check_error(LIBINFINIOP.infiniopTriu(
            descriptor, 
            None, 0, 
            output.data(), 
            input_tensor.data(), 
            None
        ))
    
    lib_triu()
    
    if sync is not None:
        sync()
    
    # 验证结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output_torch, atol=atol, rtol=rtol)
    assert torch.allclose(output.actual_tensor(), output_torch, atol=atol, rtol=rtol)
    
    # 性能分析
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: triu_torch(input_tensor.torch_tensor(), diagonal), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_triu(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    check_error(LIBINFINIOP.infiniopDestroyTriuDescriptor(descriptor))





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