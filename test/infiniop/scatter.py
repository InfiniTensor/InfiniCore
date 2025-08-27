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



# Test cases: (input_shape, index_shape, output_shape, dim, input_strides, output_strides, index_strides)
_TEST_CASES = [
    # input_shape, index_shape, output_shape, dim, input_strides, output_strides, index_strides
    # 基础2D测试用例
    ((4, 6), (3, 6), (4, 6), 0, None, None, None),  # 在第0维scatter
    ((5, 8), (5, 4), (5, 8), 1, None, None, None),  # 在第1维scatter
    
    # 3D张量测试用例
    ((3, 4, 5), (2, 4, 5), (3, 4, 5), 0, None, None, None),  # 3D在第0维
    ((4, 3, 6), (4, 2, 6), (4, 3, 6), 1, None, None, None),  # 3D在第1维
    ((2, 3, 7), (2, 3, 4), (2, 3, 7), 2, None, None, None),  # 3D在第2维
    
    # 4D张量测试用例
    ((2, 3, 4, 5), (1, 3, 4, 5), (2, 3, 4, 5), 0, None, None, None),  # 4D在第0维
    ((3, 2, 4, 6), (3, 1, 4, 6), (3, 2, 4, 6), 1, None, None, None),  # 4D在第1维
    
    # 1D张量测试用例
    ((8,), (5,), (8,), 0, None, None, None),  # 1D基础测试
    ((12,), (6,), (12,), 0, None, None, None),  # 1D较大尺寸
    
    # 边界情况
    ((1, 1), (1, 1), (1, 1), 0, None, None, None),  # 最小张量
    ((1, 1), (1, 1), (1, 1), 1, None, None, None),  # 最小张量不同维度
    
    # 强制重复索引场景（核心测试非确定性）
    ((8,), (10,), (8,), 0, None, None, None),  # 1D：10个索引→8个输出位置（必重复）
    ((4, 6), (4, 8), (4, 6), 1, None, None, None),  # 2D(dim=1)：4×8=32个索引→6个输出位置（必重复）
    ((3, 4, 5), (3, 4, 7), (3, 4, 5), 2, None, None, None),  # 3D(dim=2)：3×4×7=84个索引→5个输出位置（必重复）
    
    # 无重复索引场景（用于验证确定性分支）
    ((5,), (3,), (5,), 0, None, None, None),  # 1D：3个索引→5个输出位置（无重复）
    ((4, 6), (4, 3), (4, 6), 1, None, None, None),  # 2D(dim=1)：4×3=12个索引→6个输出位置（可设计无重复）
    
    # 自定义步长测试用例
    ((5, 6), (3, 6), (5, 6), 0, (6, 1), (1, 5), None),  # 输入行优先，输出列优先
    ((4, 8), (4, 5), (4, 8), 1, (1, 4), (8, 1), None),  # 输入列优先，输出行优先
    ((3, 4, 5), (2, 4, 5), (3, 4, 5), 0, None, None, (20, 5, 1)),  # 自定义index步长
    
    # 复杂步长组合
    ((6, 4), (4, 4), (6, 4), 0, (4, 1), (1, 6), (4, 1)),  # 所有张量都有自定义步长
    ((2, 5, 3), (1, 5, 3), (2, 5, 3), 0, (15, 3, 1), (15, 1, 5), (15, 3, 1)),  # 3D复杂步长
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


def scatter_torch(input_tensor, dim, index, src):
    """PyTorch scatter参考实现"""
    output = input_tensor.clone()
    return output.scatter_(dim, index, src)


def test_stride_like_behavior():
    """测试通过构造特定index实现"步长"效果"""
    print("\n=== 测试通过index构造实现步长效果 ===")
    
    # 测试1：在1D张量上实现步长为2的效果
    target = torch.zeros(10)
    src = torch.tensor([1.0, 2.0, 3.0])
    # 构造步长为2的索引（0, 2, 4）
    index = torch.tensor([0, 2, 4])
    result = target.scatter_(0, index, src)
    expected = torch.tensor([1., 0., 2., 0., 3., 0., 0., 0., 0., 0.])
    print(f"步长2效果 - 输入: {target.tolist()}, 结果: {result.tolist()}")
    assert torch.allclose(result, expected), "步长2效果测试失败"
    
    # 测试2：在2D张量上实现不规则间隔
    target = torch.zeros(5, 3)
    src = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    # 构造不规则索引（1, 3）
    index = torch.tensor([[1, 1, 1], [3, 3, 3]])
    result = target.scatter_(0, index, src)
    print(f"不规则间隔效果 - 在位置1和3放置数据")
    print(f"结果张量:\n{result}")
    
    # 验证结果
    expected = torch.zeros(5, 3)
    expected[1] = torch.tensor([1., 2., 3.])
    expected[3] = torch.tensor([4., 5., 6.])
    assert torch.allclose(result, expected), "不规则间隔测试失败"
    
    print("✓ 所有步长效果测试通过！")


def test(
    handle, torch_device, input_shape, index_shape, output_shape, dim, input_strides, output_strides, index_strides, inplace, dtype, sync
):
    print(
        f"Testing Scatter on {InfiniDeviceNames[torch_device]} with input_shape:{input_shape} index_shape:{index_shape} output_shape:{output_shape} dim:{dim} input_strides:{input_strides} output_strides:{output_strides} index_strides:{index_strides} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace.name}"
    )

    input = TestTensor(
        input_shape,
        input_strides,
        dtype,
        torch_device,
    )
    
    # Create src tensor with same shape as index
    src = TestTensor(
        index_shape,
        index_strides if index_strides is not None else row_major_strides(index_shape),
        dtype,
        torch_device,
    )
    
    if inplace == Inplace.INPLACE:
        if input_strides != output_strides or input_shape != output_shape:
            return
        output = input
    else:
        output = TestTensor(
            output_shape,
            output_strides,
            dtype,
            torch_device,
            mode="zeros"
        )
        # Copy input to output for scatter operation
        output.torch_tensor().copy_(input.torch_tensor())

    def generate_repeated_indices(index_shape, dim, max_index, pattern="mixed"):
        """生成确定性的重复索引模式"""
        total_indices = 1
        for s in index_shape:
            total_indices *= s
        
        if pattern == "all_same":
            # 所有索引都指向同一位置
            indices = [0] * total_indices
        elif pattern == "boundary":
            # 只使用边界索引（0和max_index-1）
            indices = [0 if i % 2 == 0 else max_index - 1 for i in range(total_indices)]
        elif pattern == "no_repeat":
            # 无重复索引（如果可能）
            if total_indices <= max_index:
                indices = list(range(total_indices))
            else:
                # 如果索引数量超过目标维度，使用循环模式
                indices = [i % max_index for i in range(total_indices)]
        else:  # "mixed"
            # 混合模式：确保有重复但覆盖多个位置
            indices = []
            for i in range(total_indices):
                if i < max_index:
                    indices.append(i)
                else:
                    # 重复前面的索引
                    indices.append(i % max_index)
        
        return indices

    def get_test_index_tensor(input_shape, index_shape, output_shape, scatter_dim):
        import random
        # 固定随机种子，确保索引可复现（关键！便于定位问题）
        random.seed(42)
        torch.manual_seed(42)
        
        max_index = output_shape[scatter_dim]  # 索引范围：0到max_index-1
        total_indices = torch.prod(torch.tensor(index_shape)).item()
        
        # 根据索引数量和目标维度大小选择模式
        if total_indices > max_index * 2:
            # 索引数量远大于目标维度，使用强制重复模式
            pattern = "mixed"
        elif total_indices > max_index:
            # 索引数量略大于目标维度，使用边界重复模式
            pattern = "boundary"
        else:
            # 索引数量小于等于目标维度，尝试无重复
            pattern = "no_repeat"
        
        indices = generate_repeated_indices(index_shape, scatter_dim, max_index, pattern)
        
        # 转换为张量并reshape
        index_tensor = torch.tensor(indices, dtype=torch.int64)
        index_tensor = index_tensor.reshape(index_shape)
        
        return index_tensor
    
    torch_index = get_test_index_tensor(input_shape, index_shape, output_shape, dim).type(torch.int64)
    index = TestTensor(
        index_shape,
        index_strides if index_strides is not None else row_major_strides(index_shape),
        InfiniDtype.I64,
        torch_device,
        "manual",
        set_tensor=torch_index
    )
    
    
    # PyTorch参考实现 - scatter src到output
    if inplace == Inplace.INPLACE:
        # 对于inplace操作，需要clone来避免内存共享问题
        output.torch_tensor().scatter_(dim, index.torch_tensor(), src.torch_tensor())
    else:
        output.torch_tensor().scatter_(dim, index.torch_tensor(), src.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateScatterDescriptor(
            handle,
            ctypes.byref(descriptor),
            input.descriptor,
            output.descriptor,
            index.descriptor,
            src.descriptor,
            dim
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [output, input, index, src]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetScatterWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_scatter():
        check_error(
            LIBINFINIOP.infiniopScatter(
                descriptor,
                workspace.data(), workspace_size.value,
                output.data(),
                input.data(),
                index.data(),
                src.data(),
                None,  # stream parameter
            )
        )

    lib_scatter()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    # 对于scatter操作，当存在重复索引时，PyTorch的行为是非确定性的
    # 我们需要验证CUDA内核的结果是否是有效的可能结果之一
    def validate_scatter_result(actual, expected, src_tensor, index_tensor, dim):
        """验证scatter操作结果，正确处理重复索引的非确定性"""
        # 检查是否有重复索引
        flat_index = index_tensor.flatten()
        has_duplicates = flat_index.unique().numel() < flat_index.numel()
        
        if not has_duplicates:
            # 无重复索引场景：结果应该是确定的
            return torch.allclose(actual, expected, atol=atol, rtol=rtol)
        
        # 有重复索引场景：验证每个位置的值是否合理
        # scatter操作语义：output = input.clone(); output[index] = src
        # 所以未被scatter的位置应该保持input的原始值
        
        # 为每个输出位置收集所有可能的源值
        position_candidates = {}
        
        # 遍历所有index位置
        import numpy as np
        for idx in np.ndindex(index_tensor.shape):
            target_idx = index_tensor[idx].item()
            
            # 检查输出索引是否有效
            if 0 <= target_idx < actual.shape[dim]:
                # 获取对应的src值
                src_value = src_tensor[idx].item()
                
                # 构造输出位置的索引：关键修复！
                # scatter操作的正确语义：output[idx] = src[idx]，但是idx[dim]被替换为target_idx
                # 即：output[idx_0, ..., idx_{dim-1}, target_idx, idx_{dim+1}, ...] = src[idx_0, ..., idx_{dim-1}, idx_dim, idx_{dim+1}, ...]
                # 所以输出位置应该保持src的所有维度索引，只替换scatter维度为target_idx
                output_idx = list(idx)
                output_idx[dim] = target_idx
                output_idx = tuple(output_idx)
                
                # 收集这个输出位置的所有可能值
                if output_idx not in position_candidates:
                    position_candidates[output_idx] = []
                position_candidates[output_idx].append(src_value)
        
        # 验证每个被scatter的位置
        for output_idx, possible_values in position_candidates.items():
            actual_value = actual[output_idx].item()
            
            # 检查实际值是否在可能值列表中（考虑浮点精度）
            is_valid = False
            matched_value = None
            for pv in possible_values:
                if abs(actual_value - pv) <= atol + rtol * abs(pv):
                    is_valid = True
                    matched_value = pv
                    break
            
            if not is_valid:
                print(f"\n=== 验证失败 ===")
                print(f"位置 {output_idx}:")
                print(f"  算子输出值: {actual_value}")
                print(f"  重复索引的所有可能值: {possible_values}")
                print(f"  容差: atol={atol}, rtol={rtol}")
                for i, pv in enumerate(possible_values):
                    diff = abs(actual_value - pv)
                    threshold = atol + rtol * abs(pv)
                    print(f"  与可能值[{i}]的差异: {diff} (阈值: {threshold})")
                print(f"=================")
                return False
            
            # 输出验证成功的信息（仅在有重复索引时）
            if len(possible_values) > 1:
                print(f"✓ 位置{output_idx}: 算子选择={actual_value}, 重复索引可能值={possible_values}, 匹配={matched_value}")
        
        return True
    
    is_valid = validate_scatter_result(output.actual_tensor(), output.torch_tensor(), 
                                     src.torch_tensor(), index.torch_tensor(), dim)
    
    if not is_valid:
        print(f"\nAssertion failed for dtype {dtype}:")
        print(f"Expected: {output.torch_tensor()}")
        print(f"Actual: {output.actual_tensor()}")
        print(f"Diff: {torch.abs(output.actual_tensor() - output.torch_tensor())}")
        print(f"Max diff: {torch.max(torch.abs(output.actual_tensor() - output.torch_tensor()))}")
        print(f"Tolerance: atol={atol}, rtol={rtol}")
        print("CUDA result is not a valid scatter result!")
    
    assert is_valid         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: output.torch_tensor().scatter_(
            dim, index.torch_tensor(), input.torch_tensor()
        ), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_scatter(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyScatterDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    # 首先运行步长效果演示测试
    test_stride_like_behavior()
    
    # 将测试用例与inplace选项组合
    test_cases_with_inplace = [
        test_case + (inplace_item,)
        for test_case in _TEST_CASES
        for inplace_item in _INPLACE
    ]

    for device in get_test_devices(args):
        test_operator(device, test, test_cases_with_inplace, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")