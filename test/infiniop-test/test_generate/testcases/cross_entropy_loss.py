import gguf
import torch
import numpy as np
import torch.nn.functional as F
from typing import List
from ml_dtypes import bfloat16
from gguf import GGMLQuantizationType

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
        
    # convert input dtype to GGUF quantization type, especially for bfloat16
    def _to_gguf_dtype(self, input):
        if input.dtype == bfloat16:
            return GGMLQuantizationType.BF16
        else:
            return np_dtype_to_ggml(input.dtype)
        
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
            test_writer.gguf_key("logits"), self.logits, raw_dtype=self._to_gguf_dtype(self.logits)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("target"), self.target, raw_dtype=self._to_gguf_dtype(self.target)
        )
        
        ans = cross_entropy_loss(
            self.logits.astype(np.float64), 
            self.target.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=GGMLQuantizationType.F64
        )
        
    
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("cross_entropy_loss.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
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
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float64,
        bfloat16,
    ]
    
    for dtype in _TENSOR_DTYPES_:
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