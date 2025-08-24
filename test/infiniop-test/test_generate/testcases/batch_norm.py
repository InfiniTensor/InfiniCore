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

def batch_norm(
    input: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    init_running_mean: np.ndarray,
    init_running_var: np.ndarray,    
    momentum: float,
    eps: float,
    inplace: Inplace
):
    torch_input = torch.from_numpy(input)
    torch_weight = torch.from_numpy(weight)
    torch_bias = torch.from_numpy(bias)
    torch_init_running_mean = torch.from_numpy(init_running_mean)
    torch_init_running_var = torch.from_numpy(init_running_var)

    bn = torch.nn.BatchNorm1d(
        num_features=torch_input.shape[1],
        eps=eps,
        momentum=momentum,
        dtype=torch.float64,
    )
    bn.weight.data = torch_weight
    bn.bias.data = torch_bias
    bn.running_mean.data = torch_init_running_mean
    bn.running_var.data = torch_init_running_var
    torch_output = bn(torch_input)
    return torch_output.detach().numpy(), \
        bn.running_mean.data.detach().numpy(), \
        bn.running_var.data.detach().numpy()


class BatchNormTestCase(InfiniopTestCase):
    def __init__(
        self,
        output: np.ndarray,
        output_shape: List[int],
        output_strides: List[int],
        running_mean: np.ndarray,
        running_mean_shape: List[int],
        running_mean_strides: List[int],
        running_var: np.ndarray,
        running_var_shape: List[int],
        running_var_strides: List[int],
        input: np.ndarray,
        input_shape: List[int],
        input_strides: List[int],
        weight: np.ndarray,
        weight_shape: List[int],
        weight_strides: List[int],
        bias: np.ndarray,
        bias_shape: List[int],
        bias_strides: List[int],
        momentum: float,
        eps: float,
        inplace: Inplace
    ):
        super().__init__("batch_norm")
        self.output = output
        self.output_shape = output_shape
        self.output_strides = output_strides
        self.running_mean = running_mean
        self.running_mean_shape = running_mean_shape
        self.running_mean_strides = running_mean_strides
        self.running_var = running_var
        self.running_var_shape = running_var_shape
        self.running_var_strides = running_var_strides
        self.input = input
        self.input_shape = input_shape
        self.input_strides = input_strides
        self.weight = weight
        self.weight_shape = weight_shape
        self.weight_strides = weight_strides
        self.bias = bias
        self.bias_shape = bias_shape
        self.bias_strides = bias_strides
        self.momentum = momentum
        self.eps = eps
        self.inplace = inplace

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.output_shape)
        test_writer.add_array(test_writer.gguf_key("running_mean.shape"), self.running_mean_shape)
        test_writer.add_array(test_writer.gguf_key("running_var.shape"), self.running_var_shape)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        test_writer.add_array(test_writer.gguf_key("weight.shape"), self.weight_shape)
        test_writer.add_array(test_writer.gguf_key("bias.shape"), self.bias_shape)
        test_writer.add_float32(test_writer.gguf_key("momentum"), self.momentum)
        test_writer.add_float32(test_writer.gguf_key("eps"), self.eps)
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
            test_writer.gguf_key("running_mean.strides"),
            gguf_strides(*(self.running_mean_strides if self.running_mean_strides is not None else contiguous_gguf_strides(self.running_mean_shape)))
        )
        test_writer.add_array(
            test_writer.gguf_key("running_var.strides"),
            gguf_strides(*(self.running_var_strides if self.running_var_strides is not None else contiguous_gguf_strides(self.running_var_shape)))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_mean"), self.running_mean, raw_dtype=np_dtype_to_ggml(self.running_mean.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("running_var"), self.running_var, raw_dtype=np_dtype_to_ggml(self.running_var.dtype)
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

        ans_output, ans_running_mean, ans_running_var = batch_norm(
            self.input.astype(np.float64),
            self.weight.astype(np.float64),
            self.bias.astype(np.float64),
            self.running_mean.astype(np.float64),
            self.running_var.astype(np.float64),
            self.momentum,
            self.eps,
            self.inplace
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
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, momentum, eps, init_running_mean, init_running_var, running_mean_strides,running_var_strides
        ((13, 4, 5,), 0.1, 1e-5, 0, 1., None, None),
        ((2, 3, 4),  0.1, 1e-4, 0.5, 1., [2], [3], ),
        ((15, 16, 17,), 0.2, 1e-5, 0., 2., None, None),
        ((50, 60, 70),  0.1, 1e-4, 0.1, 1., None, None),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, momentum, eps, init_running_mean, init_running_var, running_mean_strides, running_var_strides in _TEST_CASES_:
            for inplace in _INPLACE:
                input = np.random.rand(*shape).astype(dtype)
                weight = np.random.rand(shape[1]).astype(dtype)
                bias = np.random.rand(shape[1]).astype(dtype)
                if inplace == Inplace.INPLACE:
                    output = input
                else:
                    output = np.empty(shape, dtype=dtype)          
                running_mean = np.ones([shape[1]], dtype=dtype) * init_running_mean
                running_var = np.ones([shape[1]], dtype=dtype) * init_running_var

                test_case = BatchNormTestCase(
                    output=output,
                    output_shape=shape,
                    output_strides=None,
                    running_mean=running_mean,
                    running_mean_shape=[shape[1]],
                    running_mean_strides=running_mean_strides,
                    running_var=running_var,
                    running_var_shape=[shape[1]],
                    running_var_strides=running_var_strides,
                    input=input,
                    input_shape=shape,
                    input_strides=None,
                    weight=weight,
                    weight_shape=[shape[1]],
                    weight_strides=None,
                    bias=bias,
                    bias_shape=[shape[1]],
                    bias_strides=None,
                    momentum=momentum,
                    eps=eps,
                    inplace=inplace
                )
                test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
