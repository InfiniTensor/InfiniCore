import gguf
import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16
from gguf import GGMLQuantizationType

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def batch_norm(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    momentum: float,
    eps: float,
):
    input_tensor = torch.from_numpy(input).requires_grad_(True)
    if input_tensor.dim() != 3:
        raise ValueError("Input must be 3-dimensional: (Batch, Channels, Dim)")
    
    bn = torch.nn.BatchNorm1d(
        num_features=input_tensor.shape[1],
        eps=eps,
        momentum=momentum,
        affine=True,
        track_running_stats=True,
        dtype=torch.float64
    )
    bn.weight.data = torch.from_numpy(weight)
    bn.bias.data = torch.from_numpy(bias)
    
    output = bn(input_tensor).detach().numpy()
    running_mean = bn.running_mean.numpy()
    running_var = bn.running_var.numpy()
    
    return output, running_mean, running_var


class BatchNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        weight: np.ndarray,
        shape_weight: List[int] | None,
        stride_weight: List[int] | None,
        bias: np.ndarray,
        shape_bias: List[int] | None,
        stride_bias: List[int] | None,
        output: np.ndarray,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
        running_mean: np.ndarray,
        shape_running_mean: List[int] | None,
        stride_running_mean: List[int] | None,
        running_var: np.ndarray,
        shape_running_var: List[int] | None,
        stride_running_var: List[int] | None,
        momentum: float,
        eps: float,
    ):
        super().__init__("batch_norm")
        # input
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.weight = weight
        self.shape_weight = shape_weight
        self.stride_weight = stride_weight
        self.bias = bias
        self.shape_bias = shape_bias
        self.stride_bias = stride_bias
        self.momentum = momentum
        self.eps = eps
        # output
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.running_mean = running_mean
        self.shape_running_mean = shape_running_mean
        self.stride_running_mean = stride_running_mean
        self.running_var = running_var
        self.shape_running_var = shape_running_var
        self.stride_running_var = stride_running_var
    
    # convert input dtype to GGUF quantization type, especially for bfloat16
    def _to_gguf_dtype(self, input):
        if input.dtype == bfloat16:
            return GGMLQuantizationType.BF16
        else:
            return np_dtype_to_ggml(input.dtype)
        
    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_float64(test_writer.gguf_key("momentum"), self.momentum)
        test_writer.add_float64(test_writer.gguf_key("eps"), self.eps)
        
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.shape"), self.shape_weight)
        if self.shape_bias is not None:
            test_writer.add_array(test_writer.gguf_key("bias.shape"), self.shape_bias)
        if self.shape_output is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)
        if self.shape_running_mean is not None:
            test_writer.add_array(test_writer.gguf_key("running_mean.shape"), self.shape_running_mean)
        if self.shape_running_var is not None:
            test_writer.add_array(test_writer.gguf_key("running_var.shape"), self.shape_running_var)
            
        if self.stride_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input))
        if self.stride_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*self.stride_weight))
        if self.stride_bias is not None:
            test_writer.add_array(test_writer.gguf_key("bias.strides"), gguf_strides(*self.stride_bias))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output))
        )
        test_writer.add_array(
            test_writer.gguf_key("running_mean.strides"),
            gguf_strides(*self.stride_running_mean if self.stride_running_mean is not None else contiguous_gguf_strides(self.shape_running_mean))
        )
        test_writer.add_array(
            test_writer.gguf_key("running_var.strides"),
            gguf_strides(*self.stride_running_var if self.stride_running_var is not None else contiguous_gguf_strides(self.shape_running_var))
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=self._to_gguf_dtype(self.input)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=self._to_gguf_dtype(self.weight)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("bias"), self.bias, raw_dtype=self._to_gguf_dtype(self.bias)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=self._to_gguf_dtype(self.output)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_mean"), self.running_mean, raw_dtype=self._to_gguf_dtype(self.running_mean)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_var"), self.running_var, raw_dtype=self._to_gguf_dtype(self.running_var)
        )
        
        ans_output, ans_running_mean, ans_running_var = batch_norm(
            self.input.astype(np.float64),
            self.weight.astype(np.float64),
            self.bias.astype(np.float64),
            self.momentum,
            self.eps,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_output"), ans_output, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_running_mean"), ans_running_mean, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_running_var"), ans_running_var, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("batch_norm.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, stride_input, stride_output, momentum, eps
        ((13, 2, 4), None, None, 0.1, 1e-5),
        ((13, 2, 4), None, None, 0.3, 1e-3),
        ((13, 2, 4), (10, 4, 1), (10, 4, 1), 0.1, 1e-5),
        ((4, 8, 5632), None, None, 0.1, 1e-5),
        ((4, 8, 5632), (45056, 5632, 1), (45056, 5632, 1), 0.1, 1e-5),
        ((4, 8, 5632), (45056, 5632, 1), (45056, 5632, 1), 0.05, 1e-6),
        ((16, 4, 2816), None, None, 0.1, 1e-5),
        ((16, 4, 2816), None, None, 0.05, 1e-6),
        ((16, 4, 2816), (45056, 11264, 1), (45056, 11264, 1), 0.1, 1e-5),
        ((16, 4, 2816), (90112, 11264, 1), (90112, 11264, 1), 0.1, 1e-5),
    ]
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float16,
        bfloat16,
    ]
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_input, stride_output, momentum, eps in _TEST_CASES_:
            input = np.random.rand(*shape).astype(dtype)
            output = np.empty(tuple(0 for _ in shape), dtype=dtype)
            shape_running = [shape[1],]
            running_mean = np.empty(tuple(0 for _ in shape_running), dtype=dtype)
            running_var = np.empty(tuple(0 for _ in shape_running), dtype=dtype)
            weight = np.random.rand(shape[1]).astype(dtype)
            bias = np.random.rand(shape[1]).astype(dtype)
            
            stride_weight = None
            stride_bias = None
            stride_running_mean = None
            stride_running_var = None
            input = process_zero_stride_tensor(input, stride_input)
            weight = process_zero_stride_tensor(weight, stride_weight)
            bias = process_zero_stride_tensor(bias, stride_bias)
            output = process_zero_stride_tensor(output, stride_output)
            running_mean = process_zero_stride_tensor(running_mean, stride_running_mean)
            running_var = process_zero_stride_tensor(running_var, stride_running_var)
            
            test_case = BatchNormTestCase(
                input=input,
                shape_input=shape,
                stride_input=stride_input,
                weight=weight,
                shape_weight=shape_running,
                stride_weight=stride_weight,
                bias=bias,
                shape_bias=shape_running,
                stride_bias=stride_bias,
                output=output,
                shape_output=shape,
                stride_output=stride_output,
                running_mean=running_mean,
                shape_running_mean=shape_running,
                stride_running_mean=stride_running_mean,
                running_var=running_var,
                shape_running_var=shape_running,
                stride_running_var=stride_running_var,
                momentum=momentum,
                eps=eps,
            )
            test_cases.append(test_case)
    
    test_writer.add_tests(test_cases)
    test_writer.save()
            