import gguf
import torch
import numpy as np
import torch.nn.functional as F
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def cross_entropy_loss(
    logits: np.ndarray,
    target: np.ndarray,
):
    logits_tensor = torch.from_numpy(logits).requires_grad_(True)
    target_tensor = torch.from_numpy(target)
    
    loss = F.cross_entropy(logits_tensor, target_tensor)
    return loss.detach().numpy()


def softmax(x, axis=1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


class CrossEntropyLossTestCase(InfiniopTestCase):
    def __init__(
        self,
        logits: np.ndarray,
        shape_logits: List[int] | None,
        stride_logits: List[int] | None,
        target: np.ndarray,
        shape_target: List[int] | None,
        stride_target: List[int] | None,
    ):
        super().__init__("cross_entropy_loss")
        self.logits = logits
        self.shape_logits = shape_logits
        self.stride_logits = stride_logits
        self.target = target
        self.shape_target = shape_target
        self.stride_target = stride_target
        
    def write_test(self, test_writer: InfiniopTestWriter):
        super().write_test(test_writer)
        
        if self.shape_logits is not None:
            test_writer.add_array(test_writer.gguf_key("logits.shape"), self.shape_logits)
        if self.shape_target is not None:
            test_writer.add_array(test_writer.gguf_key("target.shape"), self.shape_target)
            
        if self.stride_logits is not None:
            test_writer.add_array(test_writer.gguf_key("logits.strides"), gguf_strides(*self.stride_logits))
        if self.stride_target is not None:
            test_writer.add_array(test_writer.gguf_key("target.strides"), gguf_strides(*self.stride_target))
        
        test_writer.add_tensor(
            test_writer.gguf_key("logits"), self.logits, raw_dtype=np_dtype_to_ggml(self.logits.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("target"), self.target, raw_dtype=np_dtype_to_ggml(self.target.dtype)
        )
        
        ans = cross_entropy_loss(
            self.logits.astype(np.float64), 
            self.target.astype(np.float64),
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
    _TEST_CASES_ = [
        # shape, stride_logits, stride_target
        ((4, 2), None, None),
        ((13, 4), None, None),
        ((13, 4), (10, 1), (10, 1)),
        ((13, 4), (10, 1), None),
        ((13, 4, 4), None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
        ((16, 5632), None, None),
        ((16, 5632), (13312, 1), (13312, 1)),
        ((4, 4, 5632), None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
    ]
    
    for shape, stride_logits, stride_target in _TEST_CASES_:
        logits = np.random.rand(*shape).astype(dtype)
        target = np.random.rand(*shape).astype(dtype)
        target = softmax(target, axis=1)
        
        logits = process_zero_stride_tensor(logits, stride_logits)
        target = process_zero_stride_tensor(target, stride_target)
        
        test_case = CrossEntropyLossTestCase(
            logits=logits,
            shape_logits=shape,
            stride_logits=stride_logits,
            target=target,
            shape_target=shape,
            stride_target=stride_target,
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
        np.float32: "cross_entropy_loss_f32.gguf",
        np.float16: "cross_entropy_loss_f16.gguf",
        bfloat16: "cross_entropy_loss_bf16.gguf",
    }
    
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
