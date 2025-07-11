import gguf
import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16
from gguf import GGMLQuantizationType

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def layer_norm(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
    eps: float,
):
    input_tensor = torch.from_numpy(input).requires_grad_(True)
    weight_tensor = torch.from_numpy(weight).requires_grad_(True)
    
    input_mean = input_tensor.mean(dim=-1, keepdim=True)
    input_var = input_tensor.var(dim=-1, keepdim=True, unbiased=False)
    input_std_deviation = torch.sqrt(input_var + eps)
    input_standardization = (input_tensor - input_mean) / input_std_deviation
    
    if bias is not None:
        bias_tensor = torch.from_numpy(bias).requires_grad_(True)
        output = input_standardization * weight_tensor + bias_tensor
    else:
        output = input_standardization * weight_tensor
    
    return (
        output.detach().numpy(), 
        input_standardization.detach().numpy(), 
        input_std_deviation.detach().numpy()
    )


class LayerNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        weight: np.ndarray,
        shape_weight: List[int] | None,
        stride_weight: List[int] | None,
        bias: np.ndarray | None,
        shape_bias: List[int] | None,
        stride_bias: List[int] | None,
        output: np.ndarray,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
        input_standardization: np.ndarray,
        shape_input_standardization: List[int] | None,
        stride_input_standardization: List[int] | None,
        input_std_deviation: np.ndarray,
        shape_input_std_deviation: List[int] | None,
        stride_input_std_deviation: List[int] | None,
        eps: float,
    ):
        super().__init__("layer_norm")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.weight = weight
        self.shape_weight = shape_weight
        self.stride_weight = stride_weight
        self.bias = bias
        self.shape_bias = shape_bias
        self.stride_bias = stride_bias
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.input_standardization = input_standardization
        self.shape_input_standardization = shape_input_standardization
        self.stride_input_standardization = stride_input_standardization
        self.input_std_deviation = input_std_deviation
        self.shape_input_std_deviation = shape_input_std_deviation
        self.stride_input_std_deviation = stride_input_std_deviation
        self.eps = eps
        
    # convert input dtype to GGUF quantization type, especially for bfloat16
    def _to_gguf_dtype(self, input):
        if input.dtype == bfloat16:
            return GGMLQuantizationType.BF16
        else:
            return np_dtype_to_ggml(input.dtype)
    
    def write_test(self, test_writer: InfiniopTestWriter):
        super().write_test(test_writer)
        test_writer.add_float64(test_writer.gguf_key("eps"), self.eps)
        
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.shape"), self.shape_weight)
        if self.bias is not None and self.shape_bias is not None:
            test_writer.add_array(test_writer.gguf_key("bias.shape"), self.shape_bias)
        if self.shape_output is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)
        if self.shape_input_standardization is not None:
            test_writer.add_array(test_writer.gguf_key("input_standardization.shape"), self.shape_input_standardization)
        if self.shape_input_std_deviation is not None:
            test_writer.add_array(test_writer.gguf_key("input_std_deviation.shape"), self.shape_input_std_deviation)
            
        if self.stride_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input))
        if self.stride_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*self.stride_weight))
        if self.bias is not None and self.stride_bias is not None:
            test_writer.add_array(test_writer.gguf_key("bias.strides"), gguf_strides(*self.stride_bias))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output))
        )
        test_writer.add_array(
            test_writer.gguf_key("input_standardization.strides"),
            gguf_strides(*self.stride_input_standardization if self.stride_input_standardization is not None else contiguous_gguf_strides(self.shape_input_standardization))
        )
        test_writer.add_array(
            test_writer.gguf_key("input_std_deviation.strides"),
            gguf_strides(*self.stride_input_std_deviation if self.stride_input_std_deviation is not None else contiguous_gguf_strides(self.shape_input_std_deviation))
        )
        
        
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=self._to_gguf_dtype(self.input)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=self._to_gguf_dtype(self.weight)
        )
        if self.bias is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("bias"), self.bias, raw_dtype=self._to_gguf_dtype(self.bias)
            )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=self._to_gguf_dtype(self.output)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input_standardization"), self.input_standardization, raw_dtype=self._to_gguf_dtype(self.input_standardization)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input_std_deviation"), self.input_std_deviation, raw_dtype=self._to_gguf_dtype(self.input_std_deviation)
        )
        
        ans_output, ans_input_standardization, ans_input_std_deviation = layer_norm(
            self.input.astype(np.float64),
            self.weight.astype(np.float64),
            self.bias.astype(np.float64) if self.bias is not None else None,
            self.eps,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_output"), ans_output, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_input_standardization"), ans_input_standardization, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_input_std_deviation"), ans_input_std_deviation, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("layer_norm.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, stride_input, stride_output, has_bias, eps
        ((2, 3, 4), None, None, True, 1e-5),
        ((2, 3, 4), None, None, False, 1e-5),
        ((2, 3, 4), (10, 4, 1), (10, 4, 1), True, 1e-5),
        ((2, 3, 4), (10, 4, 1), (10, 4, 1), False, 1e-5),
        ((2, 3, 4, 5), None, None, True, 1e-5),
        ((2, 3, 4, 5), (50, 10, 5, 1), (50, 10, 5, 1), False, 1e-5),
        ((13, 4, 4), None, None, True, 1e-5),
        ((13, 4, 4), (10, 4, 1), (10, 4, 1), True, 1e-5),
        ((4, 4, 5632), None, None, True, 1e-5),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), True, 1e-5),
    ]
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float64,
        bfloat16,
    ]
    
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_input, stride_output, has_bias, eps in _TEST_CASES_:
            shape_weight = [shape[-1],]
            shape_std = shape[:-1] + (1, )
            input = np.random.rand(*shape).astype(dtype)
            weight = np.random.rand(shape[-1]).astype(dtype)
            bias = np.random.rand(shape[-1]).astype(dtype) if has_bias else None
            output = np.empty(tuple(0 for _ in shape), dtype=dtype)
            input_standardization = np.empty(tuple(0 for _ in shape), dtype=dtype)
            input_std_deviation = np.empty(tuple(0 for _ in shape_std), dtype=dtype)
            
            stride_weight = None
            stride_bias = None
            stride_input_standardization = None
            stride_input_std_deviation = None
            input = process_zero_stride_tensor(input, stride_input)
            weight = process_zero_stride_tensor(weight, stride_weight)
            bias = process_zero_stride_tensor(bias, stride_bias) if has_bias else None
            output = process_zero_stride_tensor(output, stride_output)
            input_standardization = process_zero_stride_tensor(input_standardization, stride_input_standardization)
            input_std_deviation = process_zero_stride_tensor(input_std_deviation, stride_input_std_deviation)
            
            test_case = LayerNormTestCase(
                input=input,
                shape_input=shape,
                stride_input=stride_input,
                weight=weight,
                shape_weight=shape_weight,
                stride_weight=stride_weight,
                bias=bias,
                shape_bias=shape_weight if has_bias else None,
                stride_bias=stride_weight if has_bias else None,
                output=output,
                shape_output=shape,
                stride_output=stride_output,
                input_standardization=input_standardization,
                shape_input_standardization=shape,
                stride_input_standardization=stride_input_standardization,
                input_std_deviation=input_std_deviation,
                shape_input_std_deviation=shape_std,
                stride_input_std_deviation=stride_input_std_deviation,
                eps=eps,
            )
            test_cases.append(test_case)
    
    test_writer.add_tests(test_cases)
    test_writer.save()