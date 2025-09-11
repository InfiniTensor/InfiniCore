# 直接使用qwen3_next原版的conv1d算子进行测试
import sys
sys.path.append('/home/zhujianian/workspace/zjn/transformers_dev-qwen3_next/src')

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从qwen3_next中导入原版的torch_causal_conv1d_update函数
def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    """qwen3_next原版的torch_causal_conv1d_update函数"""
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    if activation == "silu":
        out = F.silu(out[:, :, -seq_len:])
    else:
        out = out[:, :, -seq_len:]
    out = out.to(hidden_states.dtype)
    return out

def test_qwen3_next_original_conv1d():
    """使用qwen3_next的实际参数和算子进行测试"""
    
    # qwen3_next的典型参数
    B, seq_len = 1, 10
    conv_dim = 64  # 通常是key_dim * 2 + value_dim
    conv_kernel_size = 4
    
    # 创建qwen3_next风格的conv1d层
    conv1d = nn.Conv1d(
        in_channels=conv_dim,
        out_channels=conv_dim,
        bias=False,  # qwen3_next使用bias=False
        kernel_size=conv_kernel_size,
        groups=conv_dim,  # 每个通道独立卷积
        padding=conv_kernel_size - 1,  # causal padding
    )
    
    # 固定随机种子
    torch.manual_seed(42)
    conv1d.weight.data = torch.randn_like(conv1d.weight.data) * 0.1
    
    print(f"Conv1d config: in_channels={conv_dim}, kernel_size={conv_kernel_size}, groups={conv_dim}")
    print(f"Weight shape: {conv1d.weight.shape}")
    print(f"Bias: {conv1d.bias}")
    
    # === 测试Prefill模式 ===
    # 输入: [B, conv_dim, seq_len]
    mixed_qkv = torch.randn(B, conv_dim, seq_len) * 0.5
    
    # qwen3_next的prefill计算（fallback模式）
    # mixed_qkv = F.silu(self.conv1d(mixed_qkv)[:, :, :seq_len])
    conv_output = conv1d(mixed_qkv)
    prefill_output = F.silu(conv_output[:, :, :seq_len])
    
    print(f"\n=== Prefill Mode ===")
    print(f"Input shape: {mixed_qkv.shape}")
    print(f"Conv output shape: {conv_output.shape}")
    print(f"Prefill output shape: {prefill_output.shape}")
    print(f"Input first 10 values: {mixed_qkv.flatten()[:10]}")
    print(f"Weight first 10 values: {conv1d.weight.flatten()[:10]}")
    print(f"Prefill output first 10 values: {prefill_output.flatten()[:10]}")
    
    # 保存prefill测试数据
    with open("/home/zhujianian/workspace/zjn/InfiniCore/test/x.bin", "wb") as f:
        # 注意：InfiniCore期望的是padded input，所以我们需要手动padding
        # qwen3_next的conv1d有padding=kernel_size-1，我们需要模拟这个行为
        padded_input = F.pad(mixed_qkv, (conv_kernel_size - 1, 0))
        f.write(padded_input.numpy().tobytes())
    
    with open("/home/zhujianian/workspace/zjn/InfiniCore/test/conv1d_w.bin", "wb") as f:
        f.write(conv1d.weight.detach().numpy().tobytes())
    
    # 对于InfiniCore，我们需要的是没有激活函数的原始卷积输出
    raw_conv_output = conv_output[:, :, :seq_len]  # 截取到seq_len长度
    with open("/home/zhujianian/workspace/zjn/InfiniCore/test/y.bin", "wb") as f:
        f.write(raw_conv_output.detach().numpy().tobytes())
    
    # === 测试Update模式 ===
    # 模拟单步推理
    x_now = torch.randn(B, conv_dim, 1) * 0.5  # 新输入token
    conv_state = mixed_qkv[:, :, -(conv_kernel_size-1):]  # 初始状态
    
    # 使用qwen3_next原版的update函数
    update_output = torch_causal_conv1d_update(
        x_now,
        conv_state.clone(),
        conv1d.weight.squeeze(1),  # 从[C, 1, K]变成[C, K]
        bias=None,  # qwen3_next的conv1d没有bias
        activation="silu"
    )
    
    print(f"\n=== Update Mode ===")
    print(f"x_now shape: {x_now.shape}")
    print(f"conv_state shape: {conv_state.shape}")
    print(f"Update output shape: {update_output.shape}")
    print(f"Weight squeezed shape: {conv1d.weight.squeeze(1).shape}")
    
    # 保存update测试数据
    with open("/home/zhujianian/workspace/zjn/InfiniCore/test/x_now.bin", "wb") as f:
        f.write(x_now.numpy().tobytes())
    
    with open("/home/zhujianian/workspace/zjn/InfiniCore/test/conv_state_initial.bin", "wb") as f:
        f.write(conv_state.numpy().tobytes())
    
    # 对于InfiniCore，我们需要的是没有激活函数的原始输出
    # 手动计算不带激活函数的update输出
    hidden_states_new = torch.cat([conv_state, x_now], dim=-1)
    raw_update_output = F.conv1d(hidden_states_new, conv1d.weight.squeeze(1).unsqueeze(1), bias=None, padding=0, groups=conv_dim)
    
    with open("/home/zhujianian/workspace/zjn/InfiniCore/test/y_update_pytorch.bin", "wb") as f:
        f.write(raw_update_output.detach().numpy().tobytes())
    
    print(f"\n=== Test Data Generated ===")
    print(f"Dimensions: B={B}, C={conv_dim}, L={seq_len}, K={conv_kernel_size}")
    print(f"Padded input length: {seq_len + conv_kernel_size - 1}")
    print("Files saved:")
    print("- x.bin: padded input for prefill")
    print("- conv1d_w.bin: conv1d weights")
    print("- y.bin: expected prefill output (no activation)")
    print("- x_now.bin: single token input for update")
    print("- conv_state_initial.bin: initial conv state")
    print("- y_update_pytorch.bin: expected update output (no activation)")

if __name__ == "__main__":
    test_qwen3_next_original_conv1d()