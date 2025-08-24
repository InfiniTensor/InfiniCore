from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch
from enum import Enum, auto
class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

def layer_norm_backward(
    input: np.ndarray,
    grad_output: np.ndarray,
    weight: np.ndarray,
    input_standardization: np.ndarray,
    input_std_deviation: np.ndarray,
    bias_exist: bool,
    inplace: Inplace
):
    torch_input = torch.from_numpy(input)
    torch_grad_output = torch.from_numpy(grad_output)
    torch_weight = torch.from_numpy(weight)

    ln = torch.nn.LayerNorm(
        normalized_shape=torch_input.shape[-1],
        eps=0,
        bias=bias_exist, 
        dtype=torch.float64)
    ln.weight.data = torch_weight
    torch_input.requires_grad_(True)
    torch_output = ln(torch_input)
    torch_output.backward(torch_grad_output)
    return torch_input.grad.detach().numpy(), \
            ln.weight.grad.detach().numpy(), \
            (ln.bias.grad.detach().numpy() if bias_exist else np.empty([], dtype=np.float64))


class LayerNormBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        grad_input: np.ndarray,
        grad_input_shape: List[int],
        grad_input_strides: List[int],
        grad_weight: np.ndarray,
        grad_weight_shape: List[int],
        grad_weight_strides: List[int],
        grad_bias: np.ndarray,
        grad_bias_shape: List[int],
        grad_bias_strides: List[int],
        grad_output: np.ndarray,
        grad_output_shape: List[int],
        grad_output_strides: List[int],
        weight: np.ndarray,
        weight_shape: List[int],
        weight_strides: List[int],
        input_standardization: np.ndarray,
        input_standardization_shape: List[int],
        input_standardization_strides: List[int],
        input_std_deviation: np.ndarray,
        input_std_deviation_shape: List[int],
        input_std_deviation_strides: List[int],
        bias: np.ndarray,
        input: np.ndarray,
        bias_exist: bool,
        inplace: Inplace
    ):
        super().__init__("layer_norm_backward")
        self.grad_input = grad_input
        self.grad_input_shape = grad_input_shape
        self.grad_input_strides = grad_input_strides
        self.grad_weight = grad_weight
        self.grad_weight_shape = grad_weight_shape
        self.grad_weight_strides = grad_weight_strides
        self.grad_bias = grad_bias
        self.grad_bias_shape = grad_bias_shape
        self.grad_bias_strides = grad_bias_strides
        self.grad_output = grad_output
        self.grad_output_shape = grad_output_shape
        self.grad_output_strides = grad_output_strides
        self.weight = weight
        self.weight_shape = weight_shape
        self.weight_strides = weight_strides
        self.input_standardization = input_standardization
        self.input_standardization_shape = input_standardization_shape
        self.input_standardization_strides = input_standardization_strides
        self.input_std_deviation = input_std_deviation
        self.input_std_deviation_shape = input_std_deviation_shape
        self.input_std_deviation_strides = input_std_deviation_strides
        self.bias = bias
        self.input = input
        self.bias_exist = bias_exist
        self.inplace = inplace

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("grad_input.shape"), self.grad_input_shape)
        test_writer.add_array(test_writer.gguf_key("grad_weight.shape"), self.grad_weight_shape)
        test_writer.add_array(test_writer.gguf_key("grad_bias.shape"), self.grad_bias_shape)
        test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.grad_output_shape)
        test_writer.add_array(test_writer.gguf_key("weight.shape"), self.weight_shape)
        test_writer.add_array(test_writer.gguf_key("input_standardization.shape"), self.input_standardization_shape)
        test_writer.add_array(test_writer.gguf_key("input_std_deviation.shape"), self.input_std_deviation_shape)
        test_writer.add_bool(test_writer.gguf_key("bias_exist"), self.bias_exist)
        if self.grad_output_strides is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*self.grad_output_strides))
        if self.weight_strides is not None:
            test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*self.weight_strides))
        if self.input_standardization_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input_standardization.strides"), gguf_strides(*self.input_standardization_strides))
        if self.input_std_deviation_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input_std_deviation.strides"), gguf_strides(*self.input_std_deviation_strides))
        test_writer.add_array(
            test_writer.gguf_key("grad_input.strides"),
            gguf_strides(*(self.grad_input_strides if self.grad_input_strides is not None else contiguous_gguf_strides(self.grad_input_shape)))
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_weight.strides"),
            gguf_strides(*(self.grad_weight_strides if self.grad_weight_strides is not None else contiguous_gguf_strides(self.grad_weight_shape)))
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_bias.strides"),
            gguf_strides(*(self.grad_bias_strides if self.grad_bias_strides is not None else contiguous_gguf_strides(self.grad_bias_shape)))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"), self.grad_input, raw_dtype=np_dtype_to_ggml(self.grad_input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_weight"), self.grad_weight, raw_dtype=np_dtype_to_ggml(self.grad_weight.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_bias"), self.grad_bias, raw_dtype=np_dtype_to_ggml(self.grad_bias.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"), self.grad_output, raw_dtype=np_dtype_to_ggml(self.grad_output.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=np_dtype_to_ggml(self.weight.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input_standardization"), self.input_standardization, raw_dtype=np_dtype_to_ggml(self.input_standardization.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input_std_deviation"), self.input_std_deviation, raw_dtype=np_dtype_to_ggml(self.input_std_deviation.dtype)
        )

        ans_grad_input, ans_grad_weight, ans_grad_bias = layer_norm_backward(
            self.input.astype(np.float64),
            self.grad_output.astype(np.float64),
            self.weight.astype(np.float64),
            self.input_standardization.astype(np.float64),
            self.input_std_deviation.astype(np.float64),
            self.bias_exist,
            self.inplace
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_input"), ans_grad_input, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans_grad_weight"), ans_grad_weight, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        if self.bias_exist is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("ans_grad_bias"), ans_grad_bias, raw_dtype=gguf.GGMLQuantizationType.F64
            )
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("layer_norm_backward.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, bias_exist, eps, grad_input_strides, grad_output_strides, grad_weight_strides
        ((13, 4, 4), True, [30, 4, 1], [50, 4, 1], [2]),
        ((13, 4, 4), False, [30, 4, 1], [50, 4, 1], [2]),
        ((16, 5, 5632), True, None, None, None),
        ((4, 4, 5632), False, None, None, None),
        ((40, 40, 56), True, [3600, 56, 1], None, None),        
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, bias_exist, grad_input_strides, grad_output_strides, grad_weight_strides in _TEST_CASES_:
            for inplace in _INPLACE:
                input = np.random.rand(*shape).astype(dtype)
                grad_output = np.random.rand(*shape).astype(dtype)
                weight = np.random.rand(*shape[-1:]).astype(dtype)
                mean = np.mean(input, axis=2, keepdims=True).astype(dtype)
                std = np.sqrt(np.var(input, axis=2, ddof=0)).astype(dtype)
                input_standardization = (input - mean) / np.expand_dims(std, axis=2)
                input_std_deviation = std
                if inplace == Inplace.INPLACE:
                    grad_input = grad_output
                else:
                    grad_input = np.empty(shape, dtype=dtype)              
                grad_weight = np.empty(shape[-1:], dtype=dtype)
                if bias_exist:
                    bias = np.random.rand(*shape[-1:]).astype(dtype)
                else:
                    bias = np.empty([], dtype=dtype)
                grad_bias = np.empty(shape[-1:], dtype=dtype)

                test_case = LayerNormBackwardTestCase(
                    grad_input=grad_input,
                    grad_input_shape=shape,
                    grad_input_strides=grad_input_strides,
                    grad_weight=grad_weight,
                    grad_weight_shape=shape[-1:],
                    grad_weight_strides=grad_weight_strides,
                    grad_bias=grad_bias,
                    grad_bias_shape=shape[-1:],
                    grad_bias_strides=None,
                    grad_output=grad_output,
                    grad_output_shape=shape,
                    grad_output_strides=grad_output_strides,
                    weight=weight,
                    weight_shape=shape[-1:],
                    weight_strides=None,
                    input_standardization=input_standardization,
                    input_standardization_shape=shape,
                    input_standardization_strides=None,
                    input_std_deviation=input_std_deviation,
                    input_std_deviation_shape=shape[:-1],
                    input_std_deviation_strides=None,
                    bias=bias,
                    input=input,
                    bias_exist=bias_exist,
                    inplace=inplace
                )
                test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
