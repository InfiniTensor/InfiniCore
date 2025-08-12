import gguf
import torch
import numpy as np
import torch.nn.functional as F
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor

def flash_attention_backward(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    grad_out: np.ndarray,
    mask: np.ndarray | None,
    mask_type: int,
):
    # mask
    if mask_type == 0:
        mask = None
    elif mask_type == 1:
        mask = mask
    elif mask_type == 2:
        mask = np.triu(np.ones((q.shape[-3], k.shape[-3]), dtype=np.float32), k=1)
        mask = np.where(mask == 1, -np.inf, mask)
    
    q_tensor = torch.tensor(q, requires_grad=True)
    k_tensor = torch.tensor(k, requires_grad=True)
    v_tensor = torch.tensor(v, requires_grad=True)
    grad_out_tensor = torch.tensor(grad_out)
    mask_tensor = torch.tensor(mask) if mask is not None else None
    
    if q_tensor.dim() == 3:
        # from (seq_len, num_heads, dim) to (num_heads, seq_len, dim)
        q_shaped = q_tensor.permute(1, 0, 2)
        k_shaped = k_tensor.permute(1, 0, 2)
        v_shaped = v_tensor.permute(1, 0, 2)
        grad_out_shaped = grad_out_tensor.permute(1, 0, 2)
    elif q_tensor.dim() == 4:
        # from (batch_size, seq_len, num_heads, head_dim) to (batch_size, num_heads, seq_len, head_dim)
        q_shaped = q_tensor.permute(0, 2, 1, 3)
        k_shaped = k_tensor.permute(0, 2, 1, 3)
        v_shaped = v_tensor.permute(0, 2, 1, 3)
        grad_out_shaped = grad_out_tensor.permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(
        query=q_shaped,
        key=k_shaped,
        value=v_shaped,
        attn_mask=mask_tensor,
        enable_gqa=True,
    )
    out.backward(grad_out_shaped)
    
    return (
        q_tensor.grad.numpy(),
        k_tensor.grad.numpy(),
        v_tensor.grad.numpy(),
    )
    
    
class FlashAttentionBackwardTest(InfiniopTestCase):
    def __init__(
        self,
        q: np.ndarray,
        shape_q: List[int] | None,
        stride_q: List[int] | None,
        k: np.ndarray,
        shape_k: List[int] | None,
        stride_k: List[int] | None,
        v: np.ndarray,
        shape_v: List[int] | None,
        stride_v: List[int] | None,
        grad_out: np.ndarray,
        shape_grad_out: List[int] | None,
        stride_grad_out: List[int] | None,
        grad_q: np.ndarray,
        shape_grad_q: List[int] | None,
        stride_grad_q: List[int] | None,
        grad_k: np.ndarray,
        shape_grad_k: List[int] | None,
        stride_grad_k: List[int] | None,
        grad_v: np.ndarray,
        shape_grad_v: List[int] | None,
        stride_grad_v: List[int] | None,
        mask: np.ndarray | None,
        shape_mask: List[int] | None,
        stride_mask: List[int] | None,
        mask_type: int,
    ):
        super().__init__("flash_attention_backward")
        # input
        self.q = q
        self.shape_q = shape_q
        self.stride_q = stride_q
        self.k = k
        self.shape_k = shape_k
        self.stride_k = stride_k
        self.v = v
        self.shape_v = shape_v
        self.stride_v = stride_v
        self.grad_out = grad_out
        self.shape_grad_out = shape_grad_out
        self.stride_grad_out = stride_grad_out
        self.mask = mask
        self.shape_mask = shape_mask
        self.stride_mask = stride_mask
        self.mask_type = mask_type
        # output
        self.grad_q = grad_q
        self.shape_grad_q = shape_grad_q
        self.stride_grad_q = stride_grad_q
        self.grad_k = grad_k
        self.shape_grad_k = shape_grad_k
        self.stride_grad_k = stride_grad_k
        self.grad_v = grad_v
        self.shape_grad_v = shape_grad_v
        self.stride_grad_v = stride_grad_v
    
    def write_test(self, test_writer: InfiniopTestWriter):
        super().write_test(test_writer)
        test_writer.add_int32(test_writer.gguf_key("mask_type"), self.mask_type)
        
        if self.shape_q is not None:
            test_writer.add_array(test_writer.gguf_key("q.shape"), self.shape_q)
        if self.shape_k is not None:
            test_writer.add_array(test_writer.gguf_key("k.shape"), self.shape_k)
        if self.shape_v is not None:
            test_writer.add_array(test_writer.gguf_key("v.shape"), self.shape_v)
        if self.shape_mask is not None:
            test_writer.add_array(test_writer.gguf_key("mask.shape"), self.shape_mask)
        if self.shape_grad_out is not None:
            test_writer.add_array(test_writer.gguf_key("grad_out.shape"), self.shape_grad_out)
        if self.shape_grad_q is not None:
            test_writer.add_array(test_writer.gguf_key("grad_q.shape"), self.shape_grad_q)
        if self.shape_grad_k is not None:
            test_writer.add_array(test_writer.gguf_key("grad_k.shape"), self.shape_grad_k)
        if self.shape_grad_v is not None:
            test_writer.add_array(test_writer.gguf_key("grad_v.shape"), self.shape_grad_v)
            
        if self.stride_q is not None:
            test_writer.add_array(test_writer.gguf_key("q.stride"), gguf_strides(*self.stride_q))
        if self.stride_k is not None:
            test_writer.add_array(test_writer.gguf_key("k.stride"), gguf_strides(*self.stride_k))
        if self.stride_v is not None:
            test_writer.add_array(test_writer.gguf_key("v.stride"), gguf_strides(*self.stride_v))
        if self.stride_mask is not None:
            test_writer.add_array(test_writer.gguf_key("mask.stride"), gguf_strides(*self.stride_mask))
        if self.stride_grad_out is not None:
            test_writer.add_array(test_writer.gguf_key("grad_out.stride"), gguf_strides(*self.stride_grad_out))
        test_writer.add_array(
            test_writer.gguf_key("grad_q.stride"),
            gguf_strides(*self.stride_grad_q) if self.stride_grad_q is not None else contiguous_gguf_strides(self.shape_grad_q)
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_k.stride"),
            gguf_strides(*self.stride_grad_k) if self.stride_grad_k is not None else contiguous_gguf_strides(self.shape_grad_k)
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_v.stride"),
            gguf_strides(*self.stride_grad_v) if self.stride_grad_v is not None else contiguous_gguf_strides(self.shape_grad_v)
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("q"), self.q, raw_dtype=np_dtype_to_ggml(self.q.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("k"), self.k, raw_dtype=np_dtype_to_ggml(self.k.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("v"), self.v, raw_dtype=np_dtype_to_ggml(self.v.dtype)
        )
        if self.mask is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("mask"), self.mask, raw_dtype=np_dtype_to_ggml(self.mask.dtype)
            )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_out"), self.grad_out, raw_dtype=np_dtype_to_ggml(self.grad_out.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_q"), self.grad_q, raw_dtype=np_dtype_to_ggml(self.grad_q.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_k"), self.grad_k, raw_dtype=np_dtype_to_ggml(self.grad_k.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_v"), self.grad_v, raw_dtype=np_dtype_to_ggml(self.grad_v.dtype)
        )
        
        ans_grad_q, ans_grad_k, ans_grad_v = flash_attention_backward(
            self.q.astype(np.float64),
            self.k.astype(np.float64),
            self.v.astype(np.float64),
            self.grad_out.astype(np.float64),
            self.mask.astype(np.float64) if self.mask is not None else None,
            self.mask_type,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_q"), ans_grad_q, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_k"), ans_grad_k, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_v"), ans_grad_v, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        

def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []
    
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    _TEST_CASES = [
        # shape_q, shape_kv, mask_type
        ((10, 2, 4), (10, 2, 4), 0),
        ((10, 2, 4), (10, 2, 4), 1),
        ((10, 2, 4), (10, 2, 4), 2),
        ((20, 2, 4), (10, 2, 4), 0),
        ((10, 8, 4), (10, 2, 4), 1),
        ((4, 10, 2, 4), (4, 10, 2, 4), 2),
        ((4, 20, 2, 4), (4, 10, 2, 4), 0),
        ((4, 10, 8, 4), (4, 10, 2, 4), 1),
    ]
    
    for shape_q, shape_kv, mask_type in _TEST_CASES:
        q = np.random.rand(*shape_q).astype(dtype)
        k = np.random.rand(*shape_kv).astype(dtype)
        v = np.random.rand(*shape_kv).astype(dtype)
        grad_out = np.random.rand(*shape_q).astype(dtype)
        
        shape_mask = None if mask_type == 0 else (q.shape[-3], k.shape[-3])
        if mask_type == 1:
            mask = np.random.randint(0, 2, size=shape_mask).astype(np.float32)
            mask = np.where(mask == 1, -np.inf, mask)
        else:
            mask = None
        
        grad_q = np.empty(tuple(0 for _ in shape_q), dtype=dtype)
        grad_k = np.empty(tuple(0 for _ in shape_kv), dtype=dtype)
        grad_v = np.empty(tuple(0 for _ in shape_kv), dtype=dtype)
        
        stride_q = None
        stride_kv = None
        stride_grad_out = None
        stride_grad_q = None
        stride_grad_kv = None
        stride_grad_v = None
        stride_mask = None
        
        q = process_zero_stride_tensor(q, stride_q)
        k = process_zero_stride_tensor(k, stride_kv)
        v = process_zero_stride_tensor(v, stride_kv)
        grad_out = process_zero_stride_tensor(grad_out, stride_grad_out)
        grad_q = process_zero_stride_tensor(grad_q, stride_grad_q)
        grad_k = process_zero_stride_tensor(grad_k, stride_grad_kv)
        grad_v = process_zero_stride_tensor(grad_v, stride_grad_kv)
        if mask is not None:
            mask = process_zero_stride_tensor(mask, stride_mask)
        
        test_case = FlashAttentionBackwardTest(
            q=q,
            shape_q=shape_q,
            stride_q=stride_q,
            k=k,
            shape_k=shape_kv,
            stride_k=stride_kv,
            v=v,
            shape_v=shape_kv,
            stride_v=stride_kv,
            grad_out=grad_out,
            shape_grad_out=shape_q,
            stride_grad_out=stride_grad_out,
            grad_q=grad_q,
            shape_grad_q=shape_q,
            stride_grad_q=stride_grad_q,
            grad_k=grad_k,
            shape_grad_k=shape_kv,
            stride_grad_k=stride_grad_kv,
            grad_v=grad_v,
            shape_grad_v=shape_kv,
            stride_grad_v=stride_grad_kv,
            mask=mask,
            shape_mask=shape_mask,
            stride_mask=stride_mask,
            mask_type=mask_type,
        )
        test_cases.append(test_case)
    
    test_writer.add_tests(test_cases)
    test_writer.save()

if __name__ == "__main__":
    _TENSOR_DTYPES = [
        np.float32,
        np.float16,
        bfloat16,
    ]
    dtype_filename_map = {
        np.float32: "flash_attention_backward_f32.gguf",
        np.float16: "flash_attention_backward_f16.gguf",
        bfloat16: "flash_attention_backward_bf16.gguf",
    }
    
    for dtype in _TENSOR_DTYPES:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
