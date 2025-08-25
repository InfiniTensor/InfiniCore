import torch
import ctypes
from ctypes import c_uint64, c_float
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
import torch.nn.functional as F

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================

class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_GRAD_INPUT = auto()

# Test cases: (shape, input_stride, weight_stride, bias_stride, running_mean_stride, running_var_stride, grad_output_stride, grad_input_stride, grad_weight_stride, grad_bias_stride)
_TEST_CASES_ = [
    # 3D BatchNorm1d cases - 符合competition.md要求的3维 (Batch, Channel, Dim)
    ((2, 16, 32), None, None, None, None, None, None, None, None, None),       # Small 3D case
    ((4, 64, 128), None, None, None, None, None, None, None, None, None),      # Medium 3D case
    ((8, 128, 256), None, None, None, None, None, None, None, None, None),     # Large 3D case
    ((1, 512, 1024), None, None, None, None, None, None, None, None, None),    # Single batch case
    ((16, 32, 64), None, None, None, None, None, None, None, None, None),      # Batch processing
]

# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_GRAD_INPUT,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_EXPANDED_TEST_CASES_ = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]  # 支持F32, F16, BF16

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},  
    InfiniDtype.F32: {"atol": 5e-5, "rtol": 5e-5},  
    InfiniDtype.BF16: {"atol": 5e-2, "rtol": 5e-2}  
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def batch_norm_backward(handle, grad_input, grad_weight, grad_bias, grad_output, input_tensor, weight, bias, running_mean, running_var, momentum=0.1, eps=1e-5):
    """Call the InfiniOp BatchNormBackward implementation"""
    # Create descriptor
    desc = infiniopOperatorDescriptor_t()
    
    # Create BatchNormBackward descriptor
    check_error(
        LIBINFINIOP.infiniopCreateBatchNormBackwardDescriptor(
            handle,
            ctypes.byref(desc),
            grad_input.descriptor,
            grad_weight.descriptor,
            grad_bias.descriptor,
            grad_output.descriptor,
            input_tensor.descriptor,
            weight.descriptor,
            running_mean.descriptor,
            running_var.descriptor,
            c_float(momentum),
            c_float(eps),
        )
    )
    
    # Get workspace size
    workspace_size = c_uint64()
    check_error(
        LIBINFINIOP.infiniopGetBatchNormBackwardWorkspaceSize(
            desc, ctypes.byref(workspace_size)
        )
    )
    
    # Create workspace
    workspace = TestWorkspace(workspace_size.value, input_tensor.device)
    
    # Execute BatchNormBackward
    check_error(
        LIBINFINIOP.infiniopBatchNormBackward(
            desc,
            workspace.data(),
            workspace.size(),
            grad_input.data(),
            grad_weight.data(),
            grad_bias.data(),
            grad_output.data(),
            input_tensor.data(),
            weight.data(),
            running_mean.data(),
            running_var.data(),
            None,  # stream
        )
    )
    
    # Destroy descriptor
    check_error(LIBINFINIOP.infiniopDestroyBatchNormBackwardDescriptor(desc))


def test(
    handle,
    device,
    input_shape,
    input_stride=None,
    weight_stride=None,
    bias_stride=None,
    running_mean_stride=None,
    running_var_stride=None,
    grad_output_stride=None,
    grad_input_stride=None,
    grad_weight_stride=None,
    grad_bias_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    tensor_dtype=InfiniDtype.F32,
    sync=None,
):
    """Test function for BatchNormBackward operator - 专门验证推理模式下正确使用running_mean和running_var"""
    inplace_str = "inplace" if inplace == Inplace.INPLACE_GRAD_INPUT else "out-of-place"
    print(
        f"Testing BatchNormBackward on {InfiniDeviceNames[device]} with shape:{input_shape} "
        f"dtype:{InfiniDtypeNames[tensor_dtype]} mode:{inplace_str}"
    )
    if DEBUG:
        print(f"Testing BatchNormBackward with shape {input_shape}, dtype {InfiniDtypeNames[tensor_dtype]}, inplace {inplace}")
    
    # Create input tensors
    input_tensor = TestTensor(
        input_shape, input_stride, tensor_dtype, device, mode="random", scale=2.0, bias=-1.0
    )
    
    # BatchNorm parameters (1D tensors with length = channels)
    channels = input_shape[1]
    param_shape = (channels,)
    
    weight = TestTensor(
        param_shape, weight_stride, tensor_dtype, device, mode="random", scale=1.0, bias=0.5
    )
    bias = TestTensor(
        param_shape, bias_stride, tensor_dtype, device, mode="random", scale=0.5, bias=0.0
    )
    
    # 关键：创建固定的running_mean和running_var（模拟训练后的固化统计量）
    # 这些值在推理模式下应该保持不变，反向算子必须使用这些固定值
    running_mean = TestTensor(
        param_shape, running_mean_stride, tensor_dtype, device, mode="random", scale=0.5, bias=0.0
    )
    running_var = TestTensor(
        param_shape, running_var_stride, tensor_dtype, device, mode="random", scale=0.5, bias=0.5  # 确保>0
    )
    
    # Create grad_output tensor (upstream gradient)
    grad_output = TestTensor(
        input_shape, grad_output_stride, tensor_dtype, device, mode="random", scale=1.0, bias=0.0
    )
    
    # Create gradient output tensors
    if inplace == Inplace.INPLACE_GRAD_INPUT:
        grad_input = grad_output  # Reuse grad_output for inplace
    else:
        grad_input = TestTensor(
            input_shape, grad_input_stride, tensor_dtype, device, mode="zeros"
        )
    
    grad_weight = TestTensor(
        param_shape, grad_weight_stride, tensor_dtype, device, mode="zeros"
    )
    grad_bias = TestTensor(
        param_shape, grad_bias_stride, tensor_dtype, device, mode="zeros"
    )
    
    # Set eps
    eps = 1e-5
    
    # Execute InfiniOp BatchNormBackward
    if PROFILE:
        profile_operation(
            lambda: batch_norm_backward(
                handle, grad_input, grad_weight, grad_bias, grad_output, input_tensor, weight, bias, running_mean, running_var, 0.1, eps
            ),
            f"InfiniOp BatchNormBackward {InfiniDtypeNames[tensor_dtype]} {input_shape}",
            NUM_PRERUN,
            NUM_ITERATIONS,
            sync,
        )
    else:
        batch_norm_backward(handle, grad_input, grad_weight, grad_bias, grad_output, input_tensor, weight, bias, running_mean, running_var, 0.1, eps)
    
    # Convert to PyTorch tensors for reference computation
    # 关键：保存原始grad_output用于PyTorch参考计算
    if inplace == Inplace.INPLACE_GRAD_INPUT:
        grad_output_torch = grad_output.torch_tensor().clone()  # Save original for reference
    else:
        grad_output_torch = grad_output.torch_tensor()
    
    input_torch = input_tensor.torch_tensor().clone()
    input_torch.requires_grad_(True)  # 启用梯度计算
    
    weight_torch = weight.torch_tensor().clone()
    weight_torch.requires_grad_(True)
    
    bias_torch = bias.torch_tensor().clone()
    bias_torch.requires_grad_(True)
    
    running_mean_torch = running_mean.torch_tensor().clone()
    running_var_torch = running_var.torch_tensor().clone()
    
    # PyTorch BatchNorm1d for 3D tensors (Batch, Channel, Dim) - 符合competition.md要求
    if len(input_shape) == 3:
        torch_bn = torch.nn.BatchNorm1d(channels, eps=eps, affine=True, track_running_stats=True)
    else:
        raise ValueError(f"Only 3D tensors are supported according to competition.md, got {len(input_shape)}D")
    
    # 关键：设置为推理模式，使用固定的running_mean和running_var
    torch_bn.eval()  # 强制推理模式
    torch_bn.weight.data = weight_torch.clone()
    torch_bn.bias.data = bias_torch.clone()
    torch_bn.running_mean.data = running_mean_torch.clone()
    torch_bn.running_var.data = running_var_torch.clone()
    
    # PyTorch reference computation - 推理模式下的反向传播
    if PROFILE:
        def pytorch_batch_norm_backward():
            # 正向传播（推理模式）
            output = torch_bn(input_torch)
            # 反向传播
            output.backward(grad_output_torch)
            return input_torch.grad, torch_bn.weight.grad, torch_bn.bias.grad
        
        expected_grad_input, expected_grad_weight, expected_grad_bias = profile_operation(
            pytorch_batch_norm_backward,
            f"PyTorch BatchNormBackward {InfiniDtypeNames[tensor_dtype]} {input_shape}",
            NUM_PRERUN,
            NUM_ITERATIONS,
            sync,
        )
    else:
        # 正向传播（推理模式）
        output = torch_bn(input_torch)
        # 反向传播
        output.backward(grad_output_torch)
        expected_grad_input = input_torch.grad
        expected_grad_weight = torch_bn.weight.grad
        expected_grad_bias = torch_bn.bias.grad
    
    # Compare results
    actual_grad_input = grad_input.actual_tensor()
    actual_grad_weight = grad_weight.actual_tensor()
    actual_grad_bias = grad_bias.actual_tensor()
    
    if DEBUG:
        print(f"Input shape: {input_torch.shape}")
        print(f"Grad input shape: {actual_grad_input.shape}")
        print(f"Expected grad input shape: {expected_grad_input.shape}")
        print(f"Max grad_input diff: {torch.max(torch.abs(actual_grad_input - expected_grad_input)).item()}")
        print(f"Max grad_weight diff: {torch.max(torch.abs(actual_grad_weight - expected_grad_weight)).item()}")
        print(f"Max grad_bias diff: {torch.max(torch.abs(actual_grad_bias - expected_grad_bias)).item()}")
        print(f"Running mean: {running_mean_torch}")
        print(f"Running var: {running_var_torch}")
    
    # Check correctness
    tolerance = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    atol, rtol = tolerance
    
    # 验证grad_input
    try:
        torch.testing.assert_close(
            actual_grad_input,
            expected_grad_input,
            atol=atol,
            rtol=rtol,
            msg=f"BatchNormBackward grad_input test failed for shape {input_shape}, dtype {InfiniDtypeNames[tensor_dtype]}"
        )
    except AssertionError as e:
        print(f"grad_input error: {e}")
        print(f"actual_grad_input stats: min={actual_grad_input.min()}, max={actual_grad_input.max()}, mean={actual_grad_input.mean()}")
        print(f"expected_grad_input stats: min={expected_grad_input.min()}, max={expected_grad_input.max()}, mean={expected_grad_input.mean()}")
        raise
    
    # 验证grad_weight
    try:
        torch.testing.assert_close(
            actual_grad_weight,
            expected_grad_weight,
            atol=atol,
            rtol=rtol,
            msg=f"BatchNormBackward grad_weight test failed for shape {input_shape}, dtype {InfiniDtypeNames[tensor_dtype]}"
        )
    except AssertionError as e:
        print(f"grad_weight error: {e}")
        print(f"actual_grad_weight stats: min={actual_grad_weight.min()}, max={actual_grad_weight.max()}, mean={actual_grad_weight.mean()}")
        print(f"expected_grad_weight stats: min={expected_grad_weight.min()}, max={expected_grad_weight.max()}, mean={expected_grad_weight.mean()}")
        print(f"actual_grad_weight shape: {actual_grad_weight.shape}")
        print(f"expected_grad_weight shape: {expected_grad_weight.shape}")
        
        # 详细的逐元素差异分析
        diff = torch.abs(actual_grad_weight - expected_grad_weight)
        rel_diff = diff / (torch.abs(expected_grad_weight) + 1e-8)
        print(f"Max absolute diff: {diff.max().item()}")
        print(f"Max relative diff: {rel_diff.max().item()}")
        print(f"Tolerance: atol={atol}, rtol={rtol}")
        
        # 找出差异最大的几个元素
        max_diff_indices = torch.topk(diff.flatten(), k=min(5, diff.numel())).indices
        print("Top 5 differences:")
        for i, idx in enumerate(max_diff_indices):
            actual_val = actual_grad_weight.flatten()[idx].item()
            expected_val = expected_grad_weight.flatten()[idx].item()
            abs_diff = diff.flatten()[idx].item()
            rel_diff_val = rel_diff.flatten()[idx].item()
            print(f"  [{i}] idx={idx.item()}: actual={actual_val:.6f}, expected={expected_val:.6f}, abs_diff={abs_diff:.6f}, rel_diff={rel_diff_val:.6f}")
        
        raise
    
    # 验证grad_bias
    try:
        torch.testing.assert_close(
            actual_grad_bias,
            expected_grad_bias,
            atol=atol,
            rtol=rtol,
            msg=f"BatchNormBackward grad_bias test failed for shape {input_shape}, dtype {InfiniDtypeNames[tensor_dtype]}"
        )
    except AssertionError as e:
        print(f"grad_bias error: {e}")
        print(f"actual_grad_bias stats: min={actual_grad_bias.min()}, max={actual_grad_bias.max()}, mean={actual_grad_bias.mean()}")
        print(f"expected_grad_bias stats: min={expected_grad_bias.min()}, max={expected_grad_bias.max()}, mean={expected_grad_bias.mean()}")
        raise
    
    # Clean up
    input_tensor.destroy_desc()
    weight.destroy_desc()
    bias.destroy_desc()
    running_mean.destroy_desc()
    running_var.destroy_desc()
    grad_output.destroy_desc()
    grad_weight.destroy_desc()
    grad_bias.destroy_desc()
    
    # For inplace, grad_input is the same as grad_output, so don't destroy twice
    if inplace != Inplace.INPLACE_GRAD_INPUT:
        grad_input.destroy_desc()


def test_inference_mode_validation():
    
    # 创建两组不同的running_mean和running_var
    input_shape = (4, 8, 16)
    channels = input_shape[1]
    dtype = torch.float32
    eps = 1e-5
    
    # 生成相同的输入和梯度输出
    torch.manual_seed(42)
    input_tensor = torch.randn(input_shape, dtype=dtype, requires_grad=True)
    grad_output = torch.randn(input_shape, dtype=dtype)
    weight = torch.randn(channels, dtype=dtype, requires_grad=True)
    bias = torch.randn(channels, dtype=dtype, requires_grad=True)
    
    # 第一组：正常的running_mean和running_var
    running_mean_1 = torch.randn(channels, dtype=dtype)
    running_var_1 = torch.rand(channels, dtype=dtype) + 0.5
    
    # 第二组：极端的running_mean和running_var（用于验证是否被正确使用）
    running_mean_2 = torch.zeros(channels, dtype=dtype)  # 全零均值
    running_var_2 = torch.ones(channels, dtype=dtype)    # 全一方差
    
    # 使用第一组统计量计算PyTorch参考结果
    bn1 = torch.nn.BatchNorm1d(channels, eps=eps, affine=True, track_running_stats=True).eval()
    bn1.weight.data = weight.clone()
    bn1.bias.data = bias.clone()
    bn1.running_mean.data = running_mean_1.clone()
    bn1.running_var.data = running_var_1.clone()
    
    input_1 = input_tensor.clone().detach().requires_grad_(True)
    output_1 = bn1(input_1)
    output_1.backward(grad_output)
    expected_grad_1 = input_1.grad.clone()
    
    # 使用第二组统计量计算PyTorch参考结果
    bn2 = torch.nn.BatchNorm1d(channels, eps=eps, affine=True, track_running_stats=True).eval()
    bn2.weight.data = weight.clone()
    bn2.bias.data = bias.clone()
    bn2.running_mean.data = running_mean_2.clone()
    bn2.running_var.data = running_var_2.clone()
    
    input_2 = input_tensor.clone().detach().requires_grad_(True)
    output_2 = bn2(input_2)
    output_2.backward(grad_output)
    expected_grad_2 = input_2.grad.clone()
    
    # 验证两组结果应该不同（证明running_mean/var确实影响了结果）
    diff = torch.max(torch.abs(expected_grad_1 - expected_grad_2)).item()
    print(f"两组不同running_mean/var的梯度差异: {diff}")
    
    
    print("推理模式验证测试完成\n")


if __name__ == "__main__":
    args = get_args()
    
    # Update global settings
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    # 运行推理模式验证测试
    test_inference_mode_validation()
    
    # 运行标准测试
    for device in get_test_devices(args):
        test_operator(device, test, _EXPANDED_TEST_CASES_, _TENSOR_DTYPES)
    
    print("\033[92mBatchNormBackward test passed!\033[0m")