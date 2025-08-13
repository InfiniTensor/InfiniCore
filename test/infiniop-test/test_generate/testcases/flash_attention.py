import gguf
import torch
import numpy as np
import torch.nn.functional as F
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor

def flash_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
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
    mask_tensor = torch.tensor(mask) if mask is not None else None
    
    if q_tensor.dim() == 3:
        # from (seq_len, num_heads, dim) to (num_heads, seq_len, dim)
        q_shaped = q_tensor.permute(1, 0, 2)
        k_shaped = k_tensor.permute(1, 0, 2)
        v_shaped = v_tensor.permute(1, 0, 2)
    elif q_tensor.dim() == 4:
        # from (batch_size, seq_len, num_heads, head_dim) to (batch_size, num_heads, seq_len, head_dim)
        q_shaped = q_tensor.permute(0, 2, 1, 3)
        k_shaped = k_tensor.permute(0, 2, 1, 3)
        v_shaped = v_tensor.permute(0, 2, 1, 3)

    out = F.scaled_dot_product_attention(
        query=q_shaped,
        key=k_shaped,
        value=v_shaped,
        attn_mask=mask_tensor,
        enable_gqa=True,
    )
    # Permute back to original shape
    out = out.permute(1, 0, 2) if q_tensor.dim() == 3 else out.permute(0, 2, 1, 3)
    
    return out.detach().numpy()


class FlashAttentionTestCase(InfiniopTestCase):
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
        out: np.ndarray,
        shape_out: List[int] | None,
        stride_out: List[int] | None,
        l: np.ndarray,
        shape_l: List[int] | None,
        stride_l: List[int] | None,
        mask: np.ndarray | None,
        shape_mask: List[int] | None,
        stride_mask: List[int] | None,
        mask_type: int,
    ):
        super().__init__("flash_attention")
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
        self.mask = mask
        self.shape_mask = shape_mask
        self.stride_mask = stride_mask
        self.mask_type = mask_type
        # output
        self.out = out
        self.shape_out = shape_out
        self.stride_out = stride_out
        self.l = l
        self.shape_l = shape_l
        self.stride_l = stride_l
        
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
        if self.shape_out is not None:
            test_writer.add_array(test_writer.gguf_key("out.shape"), self.shape_out)
        if self.shape_l is not None:
            test_writer.add_array(test_writer.gguf_key("l.shape"), self.shape_l)
            
        if self.stride_q is not None:
            test_writer.add_array(test_writer.gguf_key("q.strides"), gguf_strides(*self.stride_q))
        if self.stride_k is not None:
            test_writer.add_array(test_writer.gguf_key("k.strides"), gguf_strides(*self.stride_k))
        if self.stride_v is not None:
            test_writer.add_array(test_writer.gguf_key("v.strides"), gguf_strides(*self.stride_v))
        if self.stride_mask is not None:
            test_writer.add_array(test_writer.gguf_key("mask.strides"), gguf_strides(*self.stride_mask))
        test_writer.add_array(
            test_writer.gguf_key("out.strides"),
            gguf_strides(*self.stride_out if self.stride_out is not None else contiguous_gguf_strides(self.shape_out))
        )
        test_writer.add_array(
            test_writer.gguf_key("l.strides"),
            gguf_strides(*self.stride_l if self.stride_l is not None else contiguous_gguf_strides(self.shape_l))
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
        test_writer.add_tensor(
            test_writer.gguf_key("mask"), self.mask, raw_dtype=np_dtype_to_ggml(self.mask.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("out"), self.out, raw_dtype=np_dtype_to_ggml(self.out.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("l"), self.l, raw_dtype=np_dtype_to_ggml(self.l.dtype)
        )
        
        ans = flash_attention(
            self.q.astype(np.float64),
            self.k.astype(np.float64),
            self.v.astype(np.float64),
            self.mask.astype(np.float64) if self.mask is not None else None,
            self.mask_type,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        

def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape_q, shape_kv, mask_type
        # inputLayout -> ((batch_size), seq_len, num_heads, head_dim)
        ((10, 2, 4), (10, 2, 4), 0),
        ((10, 2, 4), (10, 2, 4), 1),
        ((10, 2, 4), (10, 2, 4), 2),
        ((20, 2, 4), (10, 2, 4), 0),
        ((10, 8, 4), (10, 2, 4), 1),
        ((4, 10, 2, 4), (4, 10, 2, 4), 2),
        ((4, 20, 2, 4), (4, 10, 2, 4), 0),
        ((4, 10, 8, 4), (4, 10, 2, 4), 1),
    ]
    for shape_q, shape_kv, mask_type in _TEST_CASES_:
        q = np.random.rand(*shape_q).astype(dtype)
        k = np.random.rand(*shape_kv).astype(dtype)
        v = np.random.rand(*shape_kv).astype(dtype)
        
        shape_mask = (q.shape[-3], k.shape[-3])
        mask = np.random.randint(0, 2, size=shape_mask).astype(np.float32)
        mask = np.where(mask == 1, -np.inf, mask)
            
        out = np.empty(tuple(0 for _ in shape_q), dtype=dtype)
        
        shape_l = shape_q[:-1]
        l = np.empty(tuple(0 for _ in shape_l), dtype=dtype)
        
        stride_q = None
        stride_kv = None
        stride_out = None
        stride_l = None
        stride_mask = None
        
        q = process_zero_stride_tensor(q, stride_q)
        k = process_zero_stride_tensor(k, stride_kv)
        v = process_zero_stride_tensor(v, stride_kv)
        out = process_zero_stride_tensor(out, stride_out)
        l = process_zero_stride_tensor(l, stride_l)
        mask = process_zero_stride_tensor(mask, stride_mask)
        
        test_case = FlashAttentionTestCase(
            q=q,
            shape_q=shape_q,
            stride_q=stride_q,
            k=k,
            shape_k=shape_kv,
            stride_k=stride_kv,
            v=v,
            shape_v=shape_kv,
            stride_v=stride_kv,
            out=out,
            shape_out=shape_q,
            stride_out=stride_out,
            l=l,
            shape_l=shape_l,
            stride_l=stride_l,
            mask=mask,
            shape_mask=shape_mask,
            stride_mask=stride_mask,
            mask_type=mask_type,
        )
        test_cases.append(test_case)
    
    test_writer.add_tests(test_cases)
    test_writer.save()

    
if __name__ == "__main__":
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float16,
        bfloat16,
    ]
    dtype_filename_map = {
        np.float32: "flash_attention_f32.gguf",
        np.float16: "flash_attention_f16.gguf",
        bfloat16: "flash_attention_bf16.gguf",
    }
    
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
