import gguf
import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16
from gguf import GGMLQuantizationType

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def rms_norm_backward(
    x: np.ndarray,
    grad_y: np.ndarray,
    w: np.ndarray,
    eps: float,
):
    x_tensor = torch.from_numpy(x).requires_grad_(True)
    w_tensor = torch.from_numpy(w)
    
    rms_norm = torch.nn.RMSNorm(
        normalized_shape=x_tensor.shape[-1],
        eps=eps,
        elementwise_affine=True,
        dtype=torch.float64
    )
    rms_norm.weight.data = w_tensor
    
    y = rms_norm(x_tensor)
    y.backward(torch.from_numpy(grad_y))
    
    grad_x = x_tensor.grad.numpy()
    grad_w = rms_norm.weight.grad.numpy()
    
    return grad_x, grad_w


class RMSNormBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        x: np.ndarray,
        shape_x: List[int] | None,
        stride_x: List[int] | None,
        w: np.ndarray,
        shape_w: List[int] | None,
        stride_w: List[int] | None,
        grad_y: np.ndarray,
        shape_grad_y: List[int] | None,
        stride_grad_y: List[int] | None,
        grad_x: np.ndarray,
        shape_grad_x: List[int] | None,
        stride_grad_x: List[int] | None,
        grad_w: np.ndarray,
        shape_grad_w: List[int] | None,
        stride_grad_w: List[int] | None,
        eps: float,
    ):
        super().__init__("rms_norm_backward")
        self.x = x
        self.shape_x = shape_x
        self.stride_x = stride_x
        self.w = w
        self.shape_w = shape_w
        self.stride_w = stride_w
        self.grad_y = grad_y
        self.shape_grad_y = shape_grad_y
        self.stride_grad_y = stride_grad_y
        self.grad_x = grad_x
        self.shape_grad_x = shape_grad_x
        self.stride_grad_x = stride_grad_x
        self.grad_w = grad_w
        self.shape_grad_w = shape_grad_w
        self.stride_grad_w = stride_grad_w
        self.eps = eps
        
    # convert input dtype to GGUF quantization type, especially for bfloat16
    def _to_gguf_dtype(self, input):
        if input.dtype == bfloat16:
            return GGMLQuantizationType.BF16
        else:
            return np_dtype_to_ggml(input.dtype)
        
    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_float64(test_writer.gguf_key("eps"), self.eps)
        
        if self.shape_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.shape"), self.shape_x)
        if self.shape_w is not None:
            test_writer.add_array(test_writer.gguf_key("w.shape"), self.shape_w)
        if self.shape_grad_y is not None:
            test_writer.add_array(test_writer.gguf_key("grad_y.shape"), self.shape_grad_y)
        if self.shape_grad_x is not None:
            test_writer.add_array(test_writer.gguf_key("grad_x.shape"), self.shape_grad_x)
        if self.shape_grad_w is not None:
            test_writer.add_array(test_writer.gguf_key("grad_w.shape"), self.shape_grad_w)
            
        if self.stride_x is not None:
            test_writer.add_array(test_writer.gguf_key("x.strides"), gguf_strides(*self.stride_x))
        if self.stride_w is not None:
            test_writer.add_array(test_writer.gguf_key("w.strides"), gguf_strides(*self.stride_w))
        if self.stride_grad_y is not None:
            test_writer.add_array(test_writer.gguf_key("grad_y.strides"), gguf_strides(*self.stride_grad_y))
        test_writer.add_array(
            test_writer.gguf_key("grad_x.strides"),
            gguf_strides(*self.stride_grad_x if self.stride_grad_x is not None else contiguous_gguf_strides(self.shape_grad_x))
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_w.strides"),
            gguf_strides(*self.stride_grad_w if self.stride_grad_w is not None else contiguous_gguf_strides(self.shape_grad_w))
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("x"), self.x, raw_dtype=self._to_gguf_dtype(self.x)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("w"), self.w, raw_dtype=self._to_gguf_dtype(self.w)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_y"), self.grad_y, raw_dtype=self._to_gguf_dtype(self.grad_y)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_x"), self.grad_x, raw_dtype=self._to_gguf_dtype(self.grad_x)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_w"), self.grad_w, raw_dtype=self._to_gguf_dtype(self.grad_w)
        )
        
        ans_grad_x, ans_grad_w = rms_norm_backward(
            self.x.astype(np.float64),
            self.grad_y.astype(np.float64),
            self.w.astype(np.float64),
            self.eps
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_x"), ans_grad_x, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_w"), ans_grad_w, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("rms_norm_backward.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, stride_x, stride_grad_y, stride_grad_x, bias
        ((2, 3, 4), None, None, None, 1e-5),
        ((2, 3, 4), None, None, None, 1e-5),
        ((2, 3, 4), (10, 4, 1), (10, 4, 1), (10, 4, 1), 1e-5),
        ((2, 3, 4), (10, 4, 1), (10, 4, 1), (10, 4, 1), 1e-5),
        ((2, 3, 4, 5), None, None, None, 1e-5),
        ((2, 3, 4, 5), (50, 10, 5, 1), (50, 10, 5, 1), (50, 10, 5, 1), 1e-5),
        ((13, 4, 4), None, None, None, 1e-5),
        ((13, 4, 4), (10, 4, 1), (10, 4, 1), (10, 4, 1), 1e-5),
        ((4, 4, 5632), None, None, None, 1e-5),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1), 1e-5),
    ]
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float64,
        bfloat16,
    ]
    
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_x, stride_grad_y, stride_grad_x, eps in _TEST_CASES_:
            shape_weight = [shape[-1],]
            x = np.random.rand(*shape).astype(dtype)
            grad_y = np.random.rand(*shape).astype(dtype)
            w = np.random.rand(shape[-1]).astype(dtype)
            grad_x = np.empty(tuple(0 for _ in shape), dtype=dtype)
            grad_w = np.empty((shape[-1],), dtype=dtype)
            
            stride_w = None
            stride_grad_w = None
            x = process_zero_stride_tensor(x, stride_x)
            w = process_zero_stride_tensor(w, stride_w)
            grad_y = process_zero_stride_tensor(grad_y, stride_grad_y)
            grad_x = process_zero_stride_tensor(grad_x, stride_grad_x)
            
            test_case = RMSNormBackwardTestCase(
                x=x,
                shape_x=shape,
                stride_x=stride_x,
                w=w,
                shape_w=shape_weight,
                stride_w=stride_w,
                grad_y=grad_y,
                shape_grad_y=shape,
                stride_grad_y=stride_grad_y,
                grad_x=grad_x,
                shape_grad_x=shape,
                stride_grad_x=stride_grad_x,
                grad_w=grad_w,
                shape_grad_w=shape_weight,
                stride_grad_w=stride_grad_w,
                eps=eps,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()