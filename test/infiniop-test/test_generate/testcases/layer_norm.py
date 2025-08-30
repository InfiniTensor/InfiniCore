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

def layer_norm(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    eps: float,
    bias_exist: bool,
    inplace: Inplace
):
    torch_input = torch.from_numpy(input)
    torch_weight = torch.from_numpy(weight)
    if bias_exist:
        torch_bias = torch.from_numpy(bias)

    normalized_shape = input.shape[-1:]
    ln = torch.nn.LayerNorm(
        normalized_shape=normalized_shape,
        eps=eps,
        bias=bias_exist,
        dtype=torch.float64,
    )
    ln.weight.data = torch_weight
    if bias_exist:
        ln.bias.data = torch_bias
    mean = torch_input.mean(dim=-1, keepdim=True)
    var = torch_input.var(dim=-1, correction=0)
    std = torch.sqrt(var + eps)
    return ln(torch_input).detach().numpy(), ((torch_input - mean) / std.unsqueeze(2)).detach().numpy(), std.detach().numpy()
        


class LayerNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        output: np.ndarray,
        output_shape: List[int],
        output_strides: List[int],
        input_standardization: np.ndarray,
        input_standardization_shape: List[int],
        input_standardization_strides: List[int],
        input_std_deviation: np.ndarray,
        input_std_deviation_shape: List[int],
        input_std_deviation_strides: List[int],
        input: np.ndarray,
        input_shape: List[int],
        input_strides: List[int],
        weight: np.ndarray,
        weight_shape: List[int],
        weight_strides: List[int],
        bias: np.ndarray,
        bias_shape: List[int],
        bias_strides: List[int],
        eps: float,
        bias_exist: bool,
        inplace: Inplace
    ):
        super().__init__("layer_norm")
        self.output = output
        self.output_shape = output_shape
        self.output_strides = output_strides
        self.input_standardization = input_standardization
        self.input_standardization_shape = input_standardization_shape
        self.input_standardization_strides = input_standardization_strides
        self.input_std_deviation = input_std_deviation
        self.input_std_deviation_shape = input_std_deviation_shape
        self.input_std_deviation_strides = input_std_deviation_strides
        self.input = input
        self.input_shape = input_shape
        self.input_strides = input_strides
        self.weight = weight
        self.weight_shape = weight_shape
        self.weight_strides = weight_strides
        self.bias = bias
        self.bias_shape = bias_shape
        self.bias_strides = bias_strides
        self.eps = eps
        self.bias_exist = bias_exist
        self.inplace = inplace

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.output_shape)
        test_writer.add_array(test_writer.gguf_key("input_standardization.shape"), self.input_standardization_shape)
        test_writer.add_array(test_writer.gguf_key("input_std_deviation.shape"), self.input_std_deviation_shape)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        test_writer.add_array(test_writer.gguf_key("weight.shape"), self.weight_shape)
        test_writer.add_array(test_writer.gguf_key("bias.shape"), self.bias_shape)
        test_writer.add_float32(test_writer.gguf_key("eps"), self.eps)
        test_writer.add_bool(test_writer.gguf_key("bias_exist"), self.bias_exist)
        if self.input_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.input_strides))
        if self.weight_strides is not None:
            test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*self.weight_strides))
        if self.bias_strides is not None:
            test_writer.add_array(test_writer.gguf_key("bias.strides"), gguf_strides(*self.bias_strides))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*(self.output_strides if self.output_strides is not None else contiguous_gguf_strides(self.output_shape)))
        )
        test_writer.add_array(
            test_writer.gguf_key("input_standardization.strides"),
            gguf_strides(*(self.input_standardization_strides if self.input_standardization_strides is not None else contiguous_gguf_strides(self.input_standardization_shape)))
        )
        test_writer.add_array(
            test_writer.gguf_key("input_std_deviation.strides"),
            gguf_strides(*(self.input_std_deviation_strides if self.input_std_deviation_strides is not None else contiguous_gguf_strides(self.input_std_deviation_shape)))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input_standardization"), self.input_standardization, raw_dtype=np_dtype_to_ggml(self.input_standardization.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input_std_deviation"), self.input_std_deviation, raw_dtype=np_dtype_to_ggml(self.input_std_deviation.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=np_dtype_to_ggml(self.weight.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("bias"), self.bias, raw_dtype=np_dtype_to_ggml(self.bias.dtype)
        )

        ans_output, ans_input_standardization, ans_input_std_deviation = layer_norm(
            self.input.astype(np.float64),
            self.weight.astype(np.float64),
            self.bias.astype(np.float64),
            self.eps,
            self.bias_exist,
            self.inplace
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
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, bias_exist, eps, input_strides, output_strides, weight_strides
        ((13, 4, 4), True, 1e-5, [30, 4, 1], [50, 4, 1], [2]),
        ((16, 5, 5632), True, 1e-4, None, None, None),
        ((5, 16, 5632), False, 1e-5, None, None, [10]),
        ((4, 4, 5632), True, 1e-5, None, None, None),
        ((40, 40, 56), True, 1e-5, [3600, 56, 1], None, None),
        ((40, 40, 56), False, 1e-5, [3600, 56, 1], None, None),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, bias_exist, eps, input_strides, output_strides, weight_strides in _TEST_CASES_:
            for inplace in _INPLACE:
                input = np.random.rand(*shape).astype(dtype)
                weight = np.random.rand(shape[-1]).astype(dtype)
                if bias_exist:
                    bias = np.random.rand(shape[-1]).astype(dtype)
                else:
                    bias = np.empty([], dtype=dtype)
                if inplace == Inplace.INPLACE:
                    output = input
                else:
                    output = np.empty(shape, dtype=dtype)
                input_standardization = np.empty(shape, dtype=dtype)
                input_std_deviation = np.empty(shape[:-1], dtype=dtype)

                test_case = LayerNormTestCase(
                    output=output,
                    output_shape=shape,
                    output_strides=output_strides,
                    input_standardization=input_standardization,
                    input_standardization_shape=shape,
                    input_standardization_strides=None,
                    input_std_deviation=input_std_deviation,
                    input_std_deviation_shape=shape[:-1],
                    input_std_deviation_strides=None,
                    input=input,
                    input_shape=shape,
                    input_strides=input_strides,
                    weight=weight,
                    weight_shape=shape[-1:],
                    weight_strides=weight_strides,
                    bias=bias,
                    bias_shape=bias.shape,
                    bias_strides=None,
                    eps=eps,
                    bias_exist=bias_exist,
                    inplace=inplace
                )
                test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
