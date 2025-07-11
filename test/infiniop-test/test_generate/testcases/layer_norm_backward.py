import gguf
import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16
from gguf import GGMLQuantizationType

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def layer_norm_backward(
    grad_output: np.ndarray,
    weight: np.ndarray,
    input_standardization: np.ndarray,
    input_std_deviation: np.ndarray,
):
    grad_output_tensor = torch.from_numpy(grad_output).requires_grad_(True)
    weight_tensor = torch.from_numpy(weight).requires_grad_(True)
    input_standardization_tensor = torch.from_numpy(input_standardization).requires_grad_(True)
    input_std_deviation_tensor = torch.from_numpy(input_std_deviation).requires_grad_(True)
    
    grad_weight_tensor = (grad_output_tensor * input_standardization_tensor).sum(tuple(range(grad_output_tensor.dim() - 1)))
    grad_bias_tensor = grad_output_tensor.sum(tuple(range(grad_output_tensor.dim() - 1)))
    grad_y = grad_output_tensor * weight_tensor
    mean_grad_y = grad_y.mean(dim=-1, keepdim=True)
    mean_grad_y_x_hat = (grad_y * input_standardization_tensor).mean(dim=-1, keepdim=True)
    grad_input_tensor = (grad_y - mean_grad_y - input_standardization_tensor * mean_grad_y_x_hat) / input_std_deviation_tensor
    
    return (
        grad_input_tensor.detach().numpy(),
        grad_weight_tensor.detach().numpy(),
        grad_bias_tensor.detach().numpy(),
    )


class LayerNormBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        grad_output: np.ndarray,
        shape_grad_output: List[int] | None,
        stride_grad_output: List[int] | None,
        weight: np.ndarray,
        shape_weight: List[int] | None,
        stride_weight: List[int] | None,
        input_standardization: np.ndarray,
        shape_input_standardization: List[int] | None,
        stride_input_standardization: List[int] | None,
        input_std_deviation: np.ndarray,
        shape_input_std_deviation: List[int] | None,
        stride_input_std_deviation: List[int] | None,
        grad_input: np.ndarray,
        shape_grad_input: List[int] | None,
        stride_grad_input: List[int] | None,
        grad_weight: np.ndarray,
        shape_grad_weight: List[int] | None,
        stride_grad_weight: List[int] | None,
        grad_bias: np.ndarray | None,
        shape_grad_bias: List[int] | None,
        stride_grad_bias: List[int] | None,
    ):
        super().__init__("layer_norm_backward")
        # input
        self.grad_output = grad_output
        self.shape_grad_output = shape_grad_output
        self.stride_grad_output = stride_grad_output
        self.weight = weight
        self.shape_weight = shape_weight
        self.stride_weight = stride_weight
        self.input_standardization = input_standardization
        self.shape_input_standardization = shape_input_standardization
        self.stride_input_standardization = stride_input_standardization
        self.input_std_deviation = input_std_deviation
        self.shape_input_std_deviation = shape_input_std_deviation
        self.stride_input_std_deviation = stride_input_std_deviation
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
        
        if self.shape_grad_output is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.shape_grad_output)
        if self.shape_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.shape"), self.shape_weight)
        if self.shape_input_standardization is not None:
            test_writer.add_array(test_writer.gguf_key("input_standardization.shape"), self.shape_input_standardization)
        if self.shape_input_std_deviation is not None:
            test_writer.add_array(test_writer.gguf_key("input_std_deviation.shape"), self.shape_input_std_deviation)
        if self.shape_grad_input is not None:
            test_writer.add_array(test_writer.gguf_key("grad_input.shape"), self.shape_grad_input)
        if self.shape_grad_weight is not None:
            test_writer.add_array(test_writer.gguf_key("grad_weight.shape"), self.shape_grad_weight)
            
        if self.stride_grad_output is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*self.stride_grad_output))
        if self.stride_weight is not None:
            test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*self.stride_weight))
        if self.stride_input_standardization is not None:
            test_writer.add_array(
                test_writer.gguf_key("input_standardization.strides"), gguf_strides(*self.stride_input_standardization)
            )
        if self.stride_input_std_deviation is not None:
            test_writer.add_array(
                test_writer.gguf_key("input_std_deviation.strides"), gguf_strides(*self.stride_input_std_deviation)
            )
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
            test_writer.gguf_key("grad_output"), self.grad_output, raw_dtype=self._to_gguf_dtype(self.grad_output)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=self._to_gguf_dtype(self.weight)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input_standardization"), self.input_standardization, raw_dtype=self._to_gguf_dtype(self.input_standardization)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input_std_deviation"), self.input_std_deviation, raw_dtype=self._to_gguf_dtype(self.input_std_deviation)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"), self.grad_input, raw_dtype=self._to_gguf_dtype(self.grad_input)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_weight"), self.grad_weight, raw_dtype=self._to_gguf_dtype(self.grad_weight)
        )
        if self.grad_bias is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("grad_bias"), self.grad_bias, raw_dtype=self._to_gguf_dtype(self.grad_bias)
            )
        
        ans_grad_input, ans_grad_weight, ans_grad_bias = layer_norm_backward(
            self.grad_output.astype(np.float64),
            self.weight.astype(np.float64),
            self.input_standardization.astype(np.float64),
            self.input_std_deviation.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_input"), ans_grad_input, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_weight"), ans_grad_weight, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        if ans_grad_bias is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("ans_grad_bias"), ans_grad_bias, raw_dtype=gguf.GGMLQuantizationType.F64
            )
        

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("layer_norm_backward.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, stride_grad_output, stride_input_standardization, stride_grad_input
        ((2, 3, 4), None, None, None),
        ((2, 3, 4), None, None, None),
        ((2, 3, 4), (10, 4, 1), (10, 4, 1), (10, 4, 1)),
        ((2, 3, 4), (10, 4, 1), (10, 4, 1), (10, 4, 1)),
        ((2, 3, 4, 5), None, None, None),
        ((2, 3, 4, 5), (50, 10, 5, 1), (50, 10, 5, 1), (50, 10, 5, 1)),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (10, 4, 1), (10, 4, 1), (10, 4, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
    ]
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float64,
        bfloat16,
    ]
    
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_grad_output, stride_input_standardization, stride_grad_input in _TEST_CASES_:
            shape_weight = [shape[-1],]
            shape_std = shape[:-1] + (1, )
            grad_output = np.random.rand(*shape).astype(dtype)
            weight = np.random.rand(shape[-1]).astype(dtype)
            input_standardization = np.random.rand(*shape).astype(dtype)
            input_std_deviation = np.random.rand(*shape_std).astype(dtype)
            grad_input = np.empty(tuple(0 for _ in shape), dtype=dtype)
            grad_weight = np.empty((shape[-1],), dtype=dtype)
            grad_bias = np.empty((shape[-1],), dtype=dtype)
            
            stride_weight = None
            stride_input_std_deviation = None
            stride_grad_weight = None
            stride_grad_bias = None
            grad_output = process_zero_stride_tensor(grad_output, stride_grad_output)
            weight = process_zero_stride_tensor(weight, stride_weight)
            input_standardization = process_zero_stride_tensor(input_standardization, stride_input_standardization)
            input_std_deviation = process_zero_stride_tensor(input_std_deviation, stride_input_std_deviation)
            grad_input = process_zero_stride_tensor(grad_input, stride_grad_input)
            grad_weight = process_zero_stride_tensor(grad_weight, stride_grad_weight)
            grad_bias = process_zero_stride_tensor(grad_bias, stride_grad_bias)
            
            test_case = LayerNormBackwardTestCase(
                grad_output=grad_output,
                shape_grad_output=shape,
                stride_grad_output=stride_grad_output,
                weight=weight,
                shape_weight=shape_weight,
                stride_weight=stride_weight,
                input_standardization=input_standardization,
                shape_input_standardization=shape,
                stride_input_standardization=stride_input_standardization,
                input_std_deviation=input_std_deviation,
                shape_input_std_deviation=shape_std,
                stride_input_std_deviation=stride_input_std_deviation,
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