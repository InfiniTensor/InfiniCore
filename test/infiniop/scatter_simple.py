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

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()

_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

# 只测试连续内存布局的简单用例
_SIMPLE_TEST_CASES = [
    # (input_shape, index_shape, output_shape, dim, input_strides, output_strides, index_strides)
    ((4, 6), (3, 6), (4, 6), 0, None, None, None),  # 2D在第0维scatter
    ((5, 8), (5, 4), (5, 8), 1, None, None, None),  # 2D在第1维scatter
    ((8,), (5,), (8,), 0, None, None, None),  # 1D基础测试
    ((3, 4, 5), (2, 4, 5), (3, 4, 5), 0, None, None, None),  # 3D在第0维
]

# 只测试F32类型
_TENSOR_DTYPES = [InfiniDtype.F32]
_INDEX_DTYPES = [InfiniDtype.I32, InfiniDtype.I64]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def scatter_torch(input_tensor, dim, index, src):
    output = input_tensor.clone()
    return output.scatter_(dim, index, src)

def test(
    handle, torch_device, input_shape, index_shape, output_shape, dim, input_strides, output_strides, index_strides, inplace, dtype, sync
):
    print(f"\n=== 测试用例: input_shape={input_shape}, index_shape={index_shape}, output_shape={output_shape}, dim={dim}, dtype={InfiniDtypeNames[dtype]} ===")
    
    # 创建测试张量
    src = TestTensor(index_shape, index_strides, dtype, torch_device)  # src与index形状相同
    index = TestTensor(index_shape, index_strides, InfiniDtype.I64, torch_device)
    
    # 创建输入张量（scatter操作的基础张量）
    input_tensor = TestTensor(output_shape, output_strides, dtype, torch_device)
    
    if inplace == Inplace.INPLACE:
        if input_strides != output_strides or input_shape != output_shape:
            return
        output = input_tensor
    else:
        output = TestTensor(output_shape, output_strides, dtype, torch_device)
    
    # 初始化src数据
    src.torch_tensor().uniform_(0.1, 1.0)
    
    # 生成有效的索引
    scatter_dim_size = output_shape[dim]
    index.torch_tensor().random_(0, scatter_dim_size)
    
    # 初始化输入和输出张量
    input_tensor.torch_tensor().uniform_(0.1, 1.0)
    if inplace != Inplace.INPLACE:
        output.torch_tensor().copy_(input_tensor.torch_tensor())
    
    # 计算PyTorch参考结果
    expected_output = scatter_torch(input_tensor.torch_tensor(), dim, index.torch_tensor(), src.torch_tensor())
    
    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateScatterDescriptor(
        handle, ctypes.byref(descriptor), 
        input_tensor.descriptor, output.descriptor, index.descriptor, src.descriptor, dim
    ))
    
    # 销毁描述符以防止内核直接使用它们
    for tensor in [output, input_tensor, index, src]:
        tensor.destroy_desc()
    
    # 获取工作空间大小
    workspace_size = ctypes.c_size_t(0)
    check_error(LIBINFINIOP.infiniopGetScatterWorkspaceSize(descriptor, ctypes.byref(workspace_size)))
    
    # 创建工作空间
    workspace = TestWorkspace(workspace_size.value, torch_device)
    
    def lib_scatter():
        check_error(LIBINFINIOP.infiniopScatter(
            descriptor, workspace.data(), workspace_size.value, 
            output.data(), input_tensor.data(), index.data(), src.data()
        ))
    
    # 执行算子
    lib_scatter()
    
    # 获取容差
    tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-7, "rtol": 1e-7})
    atol, rtol = tolerance["atol"], tolerance["rtol"]
    
    print(f"Expected: {expected_output}")
    print(f"Actual: {output.actual_tensor()}")
    print(f"Diff: {torch.abs(output.actual_tensor() - expected_output)}")
    print(f"Max diff: {torch.max(torch.abs(output.actual_tensor() - expected_output))}")
    
    # 验证结果
    try:
        torch.testing.assert_close(output.actual_tensor(), expected_output, atol=atol, rtol=rtol)
        print("\033[92m✓ 测试通过!\033[0m")
    except AssertionError as e:
        print(f"\033[91m✗ 测试失败: {e}\033[0m")
        raise
    
    check_error(LIBINFINIOP.infiniopDestroyScatterDescriptor(descriptor))

if __name__ == "__main__":
    args = get_args()
    
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    # 将测试用例与inplace选项组合
    test_cases_with_inplace = [
        test_case + (inplace_item,)
        for test_case in _SIMPLE_TEST_CASES
        for inplace_item in _INPLACE
    ]

    for device in get_test_devices(args):
        test_operator(device, test, test_cases_with_inplace, _TENSOR_DTYPES)
    
    print("\033[92mAll tests passed!\033[0m")