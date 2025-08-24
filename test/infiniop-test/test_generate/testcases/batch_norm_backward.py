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

def batch_norm_backward(
    input: np.ndarray,
    grad_output: np.ndarray,
    weight: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    inplace: Inplace
):
    torch_input = torch.from_numpy(input)
    torch_grad_output = torch.from_numpy(grad_output)
    torch_weight = torch.from_numpy(weight)
    torch_running_mean = torch.from_numpy(running_mean)
    torch_running_var = torch.from_numpy(running_var)
    bn = torch.nn.BatchNorm1d(
        num_features=torch_input.shape[1],
        momentum=1,
        eps=0,
        dtype=torch.float64
    )
    bn.weight.data = torch_weight
    bn.running_mean.data = torch_running_mean
    bn.running_var.data = torch_running_var

    torch_input.requires_grad_(True)
    torch_output = bn(torch_input)

    bn.running_mean.data = torch_running_mean
    bn.running_var.data = torch_running_var

    torch_output.backward(torch_grad_output)

    return torch_input.grad.detach().numpy(),\
        bn.weight.grad.detach().numpy(), \
        bn.bias.grad.detach().numpy()


class BatchNormBackwardTestCase(InfiniopTestCase):
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
        input: np.ndarray,
        input_shape: List[int],
        input_strides: List[int],
        grad_output: np.ndarray,
        grad_output_shape: List[int],
        grad_output_strides: List[int],
        weight: np.ndarray,
        weight_shape: List[int],
        weight_strides: List[int],
        running_mean: np.ndarray,
        running_mean_shape: List[int],
        running_mean_strides: List[int],
        running_var: np.ndarray,
        running_var_shape: List[int],
        running_var_strides: List[int],
        inplace: Inplace
    ):
        super().__init__("batch_norm_backward")
        self.grad_input = grad_input
        self.grad_input_shape = grad_input_shape
        self.grad_input_strides = grad_input_strides
        self.grad_weight = grad_weight
        self.grad_weight_shape = grad_weight_shape
        self.grad_weight_strides = grad_weight_strides
        self.grad_bias = grad_bias
        self.grad_bias_shape = grad_bias_shape
        self.grad_bias_strides = grad_bias_strides
        self.input = input
        self.input_shape = input_shape
        self.input_strides = input_strides
        self.grad_output = grad_output
        self.grad_output_shape = grad_output_shape
        self.grad_output_strides = grad_output_strides
        self.weight = weight
        self.weight_shape = weight_shape
        self.weight_strides = weight_strides
        self.running_mean = running_mean
        self.running_mean_shape = running_mean_shape
        self.running_mean_strides = running_mean_strides
        self.running_var = running_var
        self.running_var_shape = running_var_shape
        self.running_var_strides = running_var_strides
        self.inplace = inplace

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("grad_input.shape"), self.grad_input_shape)
        test_writer.add_array(test_writer.gguf_key("grad_weight.shape"), self.grad_weight_shape)
        test_writer.add_array(test_writer.gguf_key("grad_bias.shape"), self.grad_bias_shape)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.grad_output_shape)
        test_writer.add_array(test_writer.gguf_key("weight.shape"), self.weight_shape)
        test_writer.add_array(test_writer.gguf_key("running_mean.shape"), self.running_mean_shape)
        test_writer.add_array(test_writer.gguf_key("running_var.shape"), self.running_var_shape)
        if self.input_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.input_strides))
        if self.grad_output_strides is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*self.grad_output_strides))
        if self.weight_strides is not None:
            test_writer.add_array(test_writer.gguf_key("weight.strides"), gguf_strides(*self.weight_strides))
        if self.running_mean_strides is not None:
            test_writer.add_array(test_writer.gguf_key("running_mean.strides"), gguf_strides(*self.running_mean_strides))
        if self.running_var_strides is not None:
            test_writer.add_array(test_writer.gguf_key("running_var.strides"), gguf_strides(*self.running_var_strides))
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
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"), self.grad_output, raw_dtype=np_dtype_to_ggml(self.grad_output.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("weight"), self.weight, raw_dtype=np_dtype_to_ggml(self.weight.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_mean"), self.running_mean, raw_dtype=np_dtype_to_ggml(self.running_mean.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_var"), self.running_var, raw_dtype=np_dtype_to_ggml(self.running_var.dtype)
        )

        ans_grad_input, ans_grad_weight, ans_grad_bias = batch_norm_backward(
            self.input.astype(np.float64),
            self.grad_output.astype(np.float64),
            self.weight.astype(np.float64),
            self.running_mean.astype(np.float64),
            self.running_var.astype(np.float64),
            self.inplace
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
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, grad_weight_strides, grad_bias_strides, running_mean_strides, running_var_strides
        ((5, 4, 3), [2], None, None, None), 
        ((10, 15, 9), None, None, [5], None), 
        ((6, 50, 15), None, [3], None, [2]), 
        ((15, 35, 4), None, None, None, None), 
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, grad_weight_strides, grad_bias_strides, running_mean_strides, running_var_strides in _TEST_CASES_:
            for inplace in _INPLACE:
                input = np.random.rand(*shape).astype(dtype)
                grad_output = np.random.rand(*shape).astype(dtype)
                weight = np.random.rand(shape[1]).astype(dtype)
                running_mean = np.random.rand(shape[1]).astype(dtype)

                reshape_input = input.transpose(1, 0, 2).reshape(shape[1], -1)
                running_mean = np.mean(reshape_input, axis=1)
                running_var = np.var(reshape_input, axis=1, ddof=0)
                if inplace == Inplace.INPLACE:
                    grad_input = grad_output
                else:
                    grad_input = np.empty(shape, dtype=dtype)
                grad_weight = np.empty([shape[1]], dtype=dtype)
                grad_bias = np.empty([shape[1]], dtype=dtype)

                test_case = BatchNormBackwardTestCase(
                    grad_input=grad_input,
                    grad_input_shape=shape,
                    grad_input_strides=None,
                    grad_weight=grad_weight,
                    grad_weight_shape=[shape[1]],
                    grad_weight_strides=grad_weight_strides,
                    grad_bias=grad_bias,
                    grad_bias_shape=[shape[1]],
                    grad_bias_strides=grad_bias_strides,
                    input=input,
                    input_shape=shape,
                    input_strides=None,
                    grad_output=grad_output,
                    grad_output_shape=shape,
                    grad_output_strides=None,
                    weight=weight,
                    weight_shape=[shape[1]],
                    weight_strides=None,
                    running_mean=running_mean,
                    running_mean_shape=[shape[1]],
                    running_mean_strides=running_mean_strides,
                    running_var=running_var,
                    running_var_shape=[shape[1]],
                    running_var_strides=running_var_strides,
                    inplace=inplace
                )
                test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
