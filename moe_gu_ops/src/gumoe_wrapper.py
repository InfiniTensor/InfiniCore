import torch
import numpy as np
import gu_moe_ops # 编译出的库

class TensorWrapper:
    """把 Torch Tensor 包装成 C++ 能读懂的简单对象"""
    def __init__(self, tensor):
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self._t = tensor # 保持引用，防止被回收
        
        self.ptr = tensor.data_ptr()
        self.shape = list(tensor.shape)
        self.device_id = tensor.device.index
        
        # 简单类型映射
        if tensor.dtype == torch.float32: self.dtype_id = 0
        elif tensor.dtype == torch.bfloat16: self.dtype_id = 1
        elif tensor.dtype == torch.int32: self.dtype_id = 2
        else: raise ValueError(f"Unsupported dtype: {tensor.dtype}")

class GuMoeBlock(torch.nn.Module):
    def __init__(self, num_experts, hidden_size, intermediate_size, top_k):
        super().__init__()
        # 初始化 C++ 对象
        self.cpp_block = gu_moe_ops.GuMoeBlock(
            num_experts,
            hidden_size,
            intermediate_size,
            0, # dtype=F32
            torch.cuda.current_device()
        )
        self.top_k = top_k

    def load_weights(self, state_dict):
        # 1. 过滤并重命名权重
        clean_weights = {}
        # 假设原始 key 是 "model.layers.0.moe..." 
        # 我们需要在 Python 里做映射，变成 "moe.gate_up_proj"
        # 这里仅作演示，具体映射逻辑看你的模型结构
        
        for k, v in state_dict.items():
            # 必须转为 Numpy 且连续
            arr = v.cpu().float().numpy()
            if not arr.flags['C_CONTIGUOUS']:
                arr = np.ascontiguousarray(arr)
            clean_weights[k] = arr
            
        # 2. 调用 C++ Loader
        self.cpp_block.load_weights(clean_weights)

    def forward(self, hidden_states, top_k_indices, top_k_values):
        # 1. 包装输入
        input_w = TensorWrapper(hidden_states)
        idx_w = TensorWrapper(top_k_indices)
        val_w = TensorWrapper(top_k_values)
        
        # 2. 预分配输出 (Output Buffer)
        output = torch.empty_like(hidden_states)
        output_w = TensorWrapper(output)
        
        # 3. 调用 C++
        self.cpp_block.forward(input_w, idx_w, val_w, output_w)
        
        return output