from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch

def reduce_max(
    input: np.ndarray,
    dim: int,
):
    torch_input = torch.from_numpy(input)
    return (torch.max(torch_input, dim, keepdim=True)[0]).detach().numpy()


class ReduceMaxTestCase(InfiniopTestCase):
    def __init__(
        self,
        output: np.ndarray,
        output_shape: List[int],
        output_strides: List[int],
        input: np.ndarray,
        input_shape: List[int],
        input_strides: List[int],
        dim: int,
    ):
        super().__init__("reduce_max")
        self.output = output
        self.output_shape = output_shape
        self.output_strides = output_strides
        self.input = input
        self.input_shape = input_shape
        self.input_strides = input_strides
        self.dim = dim

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.output_shape)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        test_writer.add_uint64(test_writer.gguf_key("dim"), self.dim)
        if self.input_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.input_strides))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*(self.output_strides if self.output_strides is not None else contiguous_gguf_strides(self.output_shape)))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )

        ans = reduce_max(
            self.input.astype(np.float64),
            self.dim,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("reduce_max.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # input_shape, output_shape, dim, input_strides, output_strides
        ((13, 4), (13, 1), 1, (4, 1), (1, 1)),
        ((13, 4), (1, 4), 0, (10, 1), (10, 1)),
        ((13, 4, 4), (13, 4, 1), 2, None, None),
        ((16, 5632), (16, 1), 1, None, None),
        ((4, 4, 5632), (1, 4, 5632), 0, None, None),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for input_shape, output_shape, dim, input_strides, output_strides in _TEST_CASES_:
            input = np.random.rand(*input_shape).astype(dtype)
            output = np.empty(output_shape, dtype=dtype)

            test_case = ReduceMaxTestCase(
                output=output,
                output_shape=output_shape,
                output_strides=output_strides,
                input=input,
                input_shape=input_shape,
                input_strides=input_strides,
                dim=dim,
            )
            test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
