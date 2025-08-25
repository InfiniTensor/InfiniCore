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

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # input_shape, weight_shape, bias_shape, output_shape
    ((1, 512), (256, 512), (256,), (1, 256)),  # 基本线性变换
    ((2, 1024), (512, 1024), (512,), (2, 512)),  # 批量处理
    ((4, 2048), (1024, 2048), (1024,), (4, 1024)),  # 更大的维度
    ((1, 768), (768, 768), (768,), (1, 768)),  # 方形权重矩阵
    ((8, 256), (128, 256), None, (8, 128)),  # 无bias情况
    ((3, 512), (1024, 512), None, (3, 1024)),  # 无bias情况，不同维度
    ((1, 1), (1, 1), (1,), (1, 1)),  # 最小情况
    ((16, 4096), (2048, 4096), (2048,), (16, 2048)),  # 大批量
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


# PyTorch implementation for linear transformation
def linear_torch(input_tensor, weight, bias=None):
    """PyTorch参考实现"""
    return torch.nn.functional.linear(input_tensor, weight, bias)


def test(
    handle,
    device,
    input_shape,
    weight_shape,
    bias_shape,
    output_shape,
    dtype=InfiniDtype.F16,
    sync=None,
):
    """测试linear算子"""
    print(
        f"Testing Linear on {InfiniDeviceNames[device]} with input_shape:{input_shape} weight_shape:{weight_shape} "
        f"bias_shape:{bias_shape} output_shape:{output_shape} dtype:{InfiniDtypeNames[dtype]}"
    )
    
    # 创建输入张量
    input_tensor = TestTensor(input_shape, None, dtype, device)
    weight_tensor = TestTensor(weight_shape, None, dtype, device)
    
    # 创建bias张量（如果需要）
    bias_tensor = None
    if bias_shape is not None:
        bias_tensor = TestTensor(bias_shape, None, dtype, device)
    
    # 创建输出张量
    output_tensor = TestTensor(output_shape, None, dtype, device, mode="zeros")

    
    # 创建算子描述符
    descriptor = infiniopOperatorDescriptor_t()
    status = LIBINFINIOP.infiniopCreateLinearDescriptor(
        handle,
        ctypes.byref(descriptor),
        input_tensor.descriptor,
        weight_tensor.descriptor,
        (bias_tensor.descriptor if bias_tensor is not None else None),
        output_tensor.descriptor,
    )
    check_error(status)
    
    # 获取工作空间大小
    workspace_size = ctypes.c_size_t()
    device_type = ctypes.c_int()
    LIBINFINIOP.infiniopGetDescriptorDeviceType(descriptor, ctypes.byref(device_type))
    status = LIBINFINIOP.infiniopGetLinearWorkspaceSize(
        descriptor, ctypes.byref(workspace_size)
    )
    check_error(status)

    # 创建工作空间
    workspace = TestWorkspace(workspace_size.value, device)

    # 执行算子
    check_error(
        LIBINFINIOP.infiniopLinear(
            descriptor,
            workspace.data(),
            workspace_size.value,
            output_tensor.data(),
            input_tensor.data(),
            weight_tensor.data(),
            bias_tensor.data() if bias_tensor is not None else None,
            None,
        )
    )

    # 获取结果
    result_tensor = output_tensor.actual_tensor()
    if result_tensor.dtype == torch.bfloat16:
        result = result_tensor.float().cpu().numpy()
    else:
        result = result_tensor.cpu().numpy()

    # PyTorch参考实现
    torch_input = input_tensor.torch_tensor()
    torch_weight = weight_tensor.torch_tensor()
    torch_bias = bias_tensor.torch_tensor() if bias_tensor is not None else None
    
    expected_tensor = linear_torch(torch_input, torch_weight, torch_bias)
    if expected_tensor.dtype == torch.bfloat16:
        expected = expected_tensor.float().cpu().numpy()
    else:
        expected = expected_tensor.cpu().numpy()

    # 比较结果
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(torch.from_numpy(result), torch.from_numpy(expected), atol=atol, rtol=rtol)

    assert result.shape == expected.shape, f"Shape mismatch: {result.shape} vs {expected.shape}"
    
    result_for_compare = result_tensor.float() if result_tensor.dtype == torch.bfloat16 else result_tensor
    expected_for_compare = expected_tensor.float() if expected_tensor.dtype == torch.bfloat16 else expected_tensor
    assert torch.allclose(
        result_for_compare, expected_for_compare, atol=atol, rtol=rtol
    ), f"Linear test failed for dtype {InfiniDtypeNames[dtype]}"

    # 性能测试
    if PROFILE:
        profile_operation(
            lambda: LIBINFINIOP.infiniopLinear(
                descriptor,
                workspace.ptr,
                workspace_size.value,
                output_tensor.ptr,
                input_tensor.ptr,
                weight_tensor.ptr,
                bias_tensor.ptr if bias_tensor is not None else None,
                sync,
            ),
            NUM_PRERUN,
            NUM_ITERATIONS,
            f"Linear {InfiniDtypeNames[dtype]} {input_shape}x{weight_shape}",
        )

    # 清理资源
    check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # 运行测试
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")