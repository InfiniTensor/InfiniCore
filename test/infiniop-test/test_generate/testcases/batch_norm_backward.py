import gguf
import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16
from gguf import GGMLQuantizationType

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def batch_norm_backward(
    input: np.ndarray,
    grad_output: np.ndarray,
    weight: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
):
    input_tensor = torch.from_numpy(input).requires_grad_(True)
    if input_tensor.dim() != 3:
        raise ValueError("Input must be 3-dimensional: (Batch, Channels, Dim)")
    
    weight_tensor = torch.from_numpy(weight).requires_grad_(True)
    running_mean_tensor = torch.from_numpy(running_mean)
    running_var_tensor = torch.from_numpy(running_var)

    
    bn = torch.nn.BatchNorm1d(
        num_features=input_tensor.shape[1],
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        dtype=torch.float64
    )
    bn.weight.data = weight_tensor
    bn.running_mean = running_mean_tensor
    bn.running_var = running_var_tensor
    
    output = bn(input_tensor)
    
    output.backward(torch.from_numpy(grad_output))
    grad_input = input_tensor.grad.numpy()
    grad_weight = bn.weight.grad.numpy()
    grad_bias = bn.bias.grad.numpy()
    
    return grad_input, grad_weight, grad_bias


class BatchNormBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        grad_output: np.ndarray,
        shape_grad_output: List[int] | None,
        stride_grad_output: List[int] | None,
        weight: np.ndarray,
        shape_weight: List[int] | None,
        stride_weight: List[int] | None,
        running_mean: np.ndarray,
        shape_running_mean: List[int] | None,
        stride_running_mean: List[int] | None,
        running_var: np.ndarray,
        shape_running_var: List[int] | None,
        stride_running_var: List[int] | None,
        grad_input: np.ndarray,
        shape_grad_input: List[int] | None,
        stride_grad_input: List[int] | None,
        grad_weight: np.ndarray,
        shape_grad_weight: List[int] | None,
        stride_grad_weight: List[int] | None,
        grad_bias: np.ndarray,
        shape_grad_bias: List[int] | None,
        stride_grad_bias: List[int] | None,
    ):
        super().__init__("batch_norm_backward")
        # input
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.grad_output = grad_output
        self.shape_grad_output = shape_grad_output
        self.stride_grad_output = stride_grad_output
        self.weight = weight
        self.shape_weight = shape_weight
        self.stride_weight = stride_weight
        self.running_mean = running_mean
        self.shape_running_mean = shape_running_mean
        self.stride_running_mean = stride_running_mean
        self.running_var = running_var
        self.shape_running_var = shape_running_var
        self.stride_running_var = stride_running_var
        # output
        self.grad_input = grad_input
        self.shape_grad_input = shape_grad_input
        self.stride_grad_input = stride_grad_input
        self.grad_weight = grad_weight
        self.shape_grad_weight = shape_grad_weight
        self.stride_grad_weight = stride_grad_weight
        self.grad_bias = grad_bias
        self.shape_grad_bias = shape_grad_bias
        self.stride_grad_bias = stride_grad_bias
    
    # convert input dtype to GGUF quantization type, especially for bfloat16
    def _to_gguf_dtype(self, input):
        if input.dtype == bfloat16:
            return GGMLQuantizationType.BF16
        else:
            return np_dtype_to_ggml(input.dtype)
    
    def write_test(self, test_writer: InfiniopTestWriter):
        super().write_test(test_writer)
        
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_grad_output is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.shape_grad_output)
        if self.shape_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.shape"), self.shape_weight)
        if self.shape_running_mean is not None:
            test_writer.add_array(test_writer.gguf_key("running_mean.shape"), self.shape_running_mean)
        if self.shape_running_var is not None:
            test_writer.add_array(test_writer.gguf_key("running_var.shape"), self.shape_running_var)
        if self.shape_grad_input is not None:
            test_writer.add_array(test_writer.gguf_key("grad_input.shape"), self.shape_grad_input)
        if self.shape_grad_weight is not None:
            test_writer.add_array(test_writer.gguf_key("grad_weight.shape"), self.shape_grad_weight)
            
        if self.stride_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input))
        if self.stride_grad_output is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*self.stride_grad_output))
        if self.stride_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*self.stride_weight))
        if self.stride_running_mean is not None:
            test_writer.add_array(test_writer.gguf_key("running_mean.strides"), gguf_strides(*self.stride_running_mean))
        if self.stride_running_var is not None:
            test_writer.add_array(test_writer.gguf_key("running_var.strides"), gguf_strides(*self.stride_running_var))
        test_writer.add_array(
            test_writer.gguf_key("grad_input.strides"),
            gguf_strides(*self.stride_grad_input if self.stride_grad_input is not None else contiguous_gguf_strides(self.shape_grad_input))
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_weight.strides"),
            gguf_strides(*self.stride_grad_weight if self.stride_grad_weight is not None else contiguous_gguf_strides(self.shape_grad_weight))
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_bias.strides"),
            gguf_strides(*self.stride_grad_bias if self.stride_grad_bias is not None else contiguous_gguf_strides(self.shape_grad_bias))
        )
            
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=self._to_gguf_dtype(self.input)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"), self.grad_output, raw_dtype=self._to_gguf_dtype(self.grad_output)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=self._to_gguf_dtype(self.weight)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_mean"), self.running_mean, raw_dtype=self._to_gguf_dtype(self.running_mean)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_var"), self.running_var, raw_dtype=self._to_gguf_dtype(self.running_var)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"), self.grad_input, raw_dtype=self._to_gguf_dtype(self.grad_input)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_weight"), self.grad_weight, raw_dtype=self._to_gguf_dtype(self.grad_weight)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_bias"), self.grad_bias, raw_dtype=self._to_gguf_dtype(self.grad_bias)
        )
            
        ans_grad_input, ans_grad_weight, ans_grad_bias = batch_norm_backward(
            self.input.astype(np.float64),
            self.grad_output.astype(np.float64),
            self.weight.astype(np.float64),
            self.running_mean.astype(np.float64),
            self.running_var.astype(np.float64),
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_input"), ans_grad_input, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_weight"), ans_grad_weight, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_bias"), ans_grad_bias, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        
        
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("batch_norm_backward.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, stride_input, stride_grad_output, stride_grad_input
        ((2, 6, 4), None, None, None),
        ((2, 6, 4), None, None, None),
        ((13, 2, 4), None, None, None),
        ((13, 2, 4), (10, 4, 1), (10, 4, 1), (10, 4, 1)),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((2, 8, 5632), None, None, None),
        ((2, 8, 5632), (22528,5632, 1), (22528, 5632, 1), (22528, 5632, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
    ]
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float16,
        bfloat16,
    ]
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_input, stride_grad_output, stride_grad_input in _TEST_CASES_:
            shape_weight = [shape[1],]
            input = np.random.rand(*shape).astype(dtype)
            grad_output = np.random.rand(*shape).astype(dtype)
            weight = np.random.rand(*shape_weight).astype(dtype)
            running_mean = np.random.rand(*shape_weight).astype(dtype)
            running_var = np.random.rand(*shape_weight).astype(dtype)
            grad_input = np.empty(tuple(0 for _ in shape), dtype=dtype)
            grad_weight = np.empty(tuple(0 for _ in shape_weight), dtype=dtype)
            grad_bias = np.empty(tuple(0 for _ in shape_weight), dtype=dtype)
            
            stride_weight = None
            stride_running_mean = None
            stride_running_var = None
            stride_grad_weight = None
            stride_grad_bias = None
            input = process_zero_stride_tensor(input, stride_input)
            grad_output = process_zero_stride_tensor(grad_output, stride_grad_output)
            weight = process_zero_stride_tensor(weight, stride_weight)
            running_mean = process_zero_stride_tensor(running_mean, stride_running_mean)
            running_var = process_zero_stride_tensor(running_var, stride_running_var)
            grad_input = process_zero_stride_tensor(grad_input, stride_grad_input)
            grad_weight = process_zero_stride_tensor(grad_weight, stride_grad_weight)
            grad_bias = process_zero_stride_tensor(grad_bias, stride_grad_bias)
            
            test_case = BatchNormBackwardTestCase(
                input=input,
                shape_input=shape,
                stride_input=stride_input,
                grad_output=grad_output,
                shape_grad_output=shape,
                stride_grad_output=stride_grad_output,
                weight=weight,
                shape_weight=shape_weight,
                stride_weight=stride_weight,
                running_mean=running_mean,
                shape_running_mean=shape_weight,
                stride_running_mean=stride_running_mean,
                running_var=running_var,
                shape_running_var=shape_weight,
                stride_running_var=stride_running_var,
                grad_input=grad_input,
                shape_grad_input=shape,
                stride_grad_input=stride_grad_input,
                grad_weight=grad_weight,
                shape_grad_weight=shape_weight,
                stride_grad_weight=stride_grad_weight,
                grad_bias=grad_bias,
                shape_grad_bias=shape_weight,
                stride_grad_bias=stride_grad_bias,
            )
            test_cases.append(test_case)
    
    test_writer.add_tests(test_cases)
    test_writer.save()