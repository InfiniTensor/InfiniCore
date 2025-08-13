from ast import List
import numpy as np
import gguf
from typing import List
import torch
import torch.nn.functional as F
from ml_dtypes import bfloat16

# 直接从transformers库导入DeepseekV3TopkRouter
from transformers.models.deepseek_v3.modular_deepseek_v3 import DeepseekV3TopkRouter

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides


class TopkRouterConfig:
    def __init__(self, hidden_size, num_experts_per_tok, n_routed_experts, routed_scaling_factor, 
                 n_group, topk_group, norm_topk_prob):
        self.hidden_size = hidden_size
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.n_group = n_group
        self.topk_group = topk_group
        self.norm_topk_prob = norm_topk_prob


def topk_router_reference(input_tensor, w_gate, config):
    """
    Reference implementation using the DeepseekV3TopkRouter from transformers
    """
    # Create router instance
    router = DeepseekV3TopkRouter(config)
    
    # Set the weight
    with torch.no_grad():
        router.weight.copy_(torch.from_numpy(w_gate).float())
    
    # Convert input to torch tensor
    input_torch = torch.from_numpy(input_tensor).float()
    
    # Forward pass
    with torch.no_grad():
        topk_indices, topk_weights = router(input_torch)
    
    return topk_indices.numpy(), topk_weights.numpy()


class TopkRouterTestCase(InfiniopTestCase):
    def __init__(
        self,
        topk_indices: np.ndarray,
        topk_weights: np.ndarray,
        input_tensor: np.ndarray,
        w_gate: np.ndarray,
        topk: int,
        shape_topk_indices: List[int] | None,
        shape_topk_weights: List[int] | None,
        shape_input: List[int] | None,
        shape_w_gate: List[int] | None,
        stride_topk_indices: List[int] | None,
        stride_topk_weights: List[int] | None,
        stride_input: List[int] | None,
        stride_w_gate: List[int] | None,
    ):
        super().__init__("topk_router")
        self.topk_indices = topk_indices
        self.topk_weights = topk_weights
        self.input_tensor = input_tensor
        self.w_gate = w_gate
        self.topk = topk
        self.shape_topk_indices = shape_topk_indices
        self.shape_topk_weights = shape_topk_weights
        self.shape_input = shape_input
        self.shape_w_gate = shape_w_gate
        self.stride_topk_indices = stride_topk_indices
        self.stride_topk_weights = stride_topk_weights
        self.stride_input = stride_input
        self.stride_w_gate = stride_w_gate

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # Write output tensors
        test_writer.add_tensor(
            test_writer.gguf_key("topk_indices"), 
            self.topk_indices, 
            raw_dtype=np_dtype_to_ggml(self.topk_indices.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("topk_weights"), 
            self.topk_weights, 
            raw_dtype=np_dtype_to_ggml(self.topk_weights.dtype)
        )
        
        # Write input tensors
        test_writer.add_tensor(
            test_writer.gguf_key("input"), 
            self.input_tensor, 
            raw_dtype=np_dtype_to_ggml(self.input_tensor.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("w_gate"), 
            self.w_gate, 
            raw_dtype=np_dtype_to_ggml(self.w_gate.dtype)
        )
        
        # Write topk parameter
        test_writer.add_array(test_writer.gguf_key("topk"), [self.topk])
        
        # Write shapes if provided
        if self.shape_topk_indices is not None:
            test_writer.add_array(test_writer.gguf_key("topk_indices.shape"), self.shape_topk_indices)
        if self.shape_topk_weights is not None:
            test_writer.add_array(test_writer.gguf_key("topk_weights.shape"), self.shape_topk_weights)
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_w_gate is not None:
            test_writer.add_array(test_writer.gguf_key("w_gate.shape"), self.shape_w_gate)
        
        # Write strides
        test_writer.add_array(
            test_writer.gguf_key("topk_indices.strides"),
            gguf_strides(*self.stride_topk_indices if self.stride_topk_indices is not None 
                        else contiguous_gguf_strides(self.shape_topk_indices))
        )
        test_writer.add_array(
            test_writer.gguf_key("topk_weights.strides"),
            gguf_strides(*self.stride_topk_weights if self.stride_topk_weights is not None 
                        else contiguous_gguf_strides(self.shape_topk_weights))
        )
        test_writer.add_array(
            test_writer.gguf_key("input.strides"),
            gguf_strides(*self.stride_input if self.stride_input is not None 
                        else contiguous_gguf_strides(self.shape_input))
        )
        test_writer.add_array(
            test_writer.gguf_key("w_gate.strides"),
            gguf_strides(*self.stride_w_gate if self.stride_w_gate is not None 
                        else contiguous_gguf_strides(self.shape_w_gate))
        )

        # Compute and write reference answer
        config = TopkRouterConfig(
            hidden_size=self.w_gate.shape[1],
            num_experts_per_tok=self.topk,
            n_routed_experts=self.w_gate.shape[0],
            routed_scaling_factor=1.0,
            n_group=min(8, self.w_gate.shape[0] // 8) if self.w_gate.shape[0] >= 64 else self.w_gate.shape[0],
            topk_group=min(3, self.topk),
            norm_topk_prob=True
        )
        
        topk_indices_ref, topk_weights_ref = topk_router_reference(
            self.input_tensor.astype(np.float64),
            self.w_gate.astype(np.float64),
            config
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("ans_topk_indices"), 
            topk_indices_ref, 
            raw_dtype=gguf.GGMLQuantizationType.I32
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_topk_weights"), 
            topk_weights_ref, 
            raw_dtype=gguf.GGMLQuantizationType.F64
        )


def gen_gguf(dtype: torch.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []
    
    _TEST_CASES_ = [
        # (batch_size, seq_len, hidden_size, n_routed_experts, topk, input_stride, w_gate_stride, topk_indices_stride, topk_weights_stride)
        (1, 32, 512, 64, 6, None, None, None, None),
        (2, 64, 1024, 128, 8, None, None, None, None),
        # 测试非连续内存布局
        (4, 128, 256, 64, 6, (512, 1), None, None, None),
        (8, 64, 512, 128, 8, None, (1024, 1), None, None),
        (2, 32, 256, 64, 6, (512, 1), (512, 1), None, None),
        # 测试输出张量的非连续布局
        (1, 64, 256, 64, 6, None, None, (12, 1), (12, 1)),
    ]

    for batch_size, seq_len, hidden_size, n_routed_experts, topk, input_stride, w_gate_stride, topk_indices_stride, topk_weights_stride in _TEST_CASES_:
        # Generate random input
        input_tensor = np.random.randn(batch_size * seq_len, hidden_size).astype(dtype)
        w_gate = np.random.randn(n_routed_experts, hidden_size).astype(dtype)
        
        # Create empty output tensors for GGUF format
        topk_indices = np.empty((0, topk), dtype=np.int32)
        topk_weights = np.empty((0, topk), dtype=dtype)
        
        test_case = TopkRouterTestCase(
            topk_indices=topk_indices,
            topk_weights=topk_weights,
            input_tensor=input_tensor,
            w_gate=w_gate,
            topk=topk,
            shape_topk_indices=[batch_size * seq_len, topk],
            shape_topk_weights=[batch_size * seq_len, topk],
            shape_input=[batch_size * seq_len, hidden_size],
            shape_w_gate=[n_routed_experts, hidden_size],
            stride_input=input_stride,
            stride_w_gate=w_gate_stride,
            stride_topk_indices=topk_indices_stride,
            stride_topk_weights=topk_weights_stride,
        )
        test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()

if __name__ == "__main__":
    _TENSOR_DTYPES_ = [np.float16, np.float32, bfloat16]
    dtype_filename_map = {
        np.float16: "topk_router_f16.gguf",
        np.float32: "topk_router_f32.gguf",
        bfloat16: "topk_router_bf16.gguf",
    }
    # 生成测试用例
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)