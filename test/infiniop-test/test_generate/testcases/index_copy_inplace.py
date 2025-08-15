from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch
import random

def index_copy_inplace(
    output: np.ndarray,
    input: np.ndarray,
    index: np.ndarray,
    dim: int,
):
    torch_output = torch.from_numpy(output)
    torch_input = torch.from_numpy(input)
    torch_index = torch.from_numpy(index)
    torch_output.index_copy_(dim, torch_index, torch_input)
    return torch_output.detach().numpy()


class IndexCopyInplaceTestCase(InfiniopTestCase):
    def __init__(
        self,
        output: np.ndarray,
        output_shape: List[int],
        output_strides: List[int],
        input: np.ndarray,
        input_shape: List[int],
        input_strides: List[int],
        index: np.ndarray,
        index_shape: List[int],
        index_strides: List[int],
        dim: int,
    ):
        super().__init__("index_copy_inplace")
        self.output = output
        self.output_shape = output_shape
        self.output_strides = output_strides
        self.input = input
        self.input_shape = input_shape
        self.input_strides = input_strides
        self.index = index
        self.index_shape = index_shape
        self.index_strides = index_strides
        self.dim = dim

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.output_shape)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        test_writer.add_array(test_writer.gguf_key("index.shape"), self.index_shape)
        test_writer.add_uint64(test_writer.gguf_key("dim"), self.dim)
        if self.input_strides is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.input_strides))
        if self.index_strides is not None:
            test_writer.add_array(test_writer.gguf_key("index.strides"), gguf_strides(*self.index_strides))
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
        test_writer.add_tensor(
            test_writer.gguf_key("index"), self.index, raw_dtype=np_dtype_to_ggml(self.index.dtype)
        )

        ans = index_copy_inplace(
            self.output,
            self.input,
            self.index,
            self.dim,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("index_copy_inplace.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # input_shape, output_shape, dim, output_strides, input_strides, index_strides,
        ([13, 1], [13, 4], 1, [37, 1], [37, 1], None),
        ([1333, 4], [1333, 4], 0, [1, 1333], [1, 2333], [2]),
        ([133, 23, 53], [133, 23, 53], 1, None, None, None),
        ([133, 23, 13, 53], [133, 23, 13, 53], 2, None, None, None),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16, np.int16, np.int32, np.bool_]
    for dtype in _TENSOR_DTYPES_:
        for input_shape, output_shape, dim, output_strides, input_strides, index_strides in _TEST_CASES_:
            input = np.random.rand(*input_shape).astype(dtype)
            
            index_list = list(range(output_shape[dim]))
            random.shuffle(index_list)
            index = np.array(index_list[:input_shape[dim]], dtype=np.int64)
            output = np.zeros(output_shape, dtype=dtype)

            test_case = IndexCopyInplaceTestCase(
                output=output,
                output_shape=output_shape,
                output_strides=output_strides,
                input=input,
                input_shape=input_shape,
                input_strides=input_strides,
                index=index,
                index_shape=[input_shape[dim]],
                index_strides=index_strides,
                dim=dim
            )
            test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
