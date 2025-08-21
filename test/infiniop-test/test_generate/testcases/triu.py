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

def triu(
    output: np.ndarray,
    input: np.ndarray,
    diagonal: int,
    inplace: Inplace
):
    torch_input = torch.from_numpy(input)
    if inplace == Inplace.INPLACE:
        torch_output = torch_input
    else:
        torch_output = torch.from_numpy(output)

    torch.triu(torch_input, diagonal, out=torch_output)

    return torch_output.detach().numpy()


class TriuTestCase(InfiniopTestCase):
    def __init__(
        self,
        output: np.ndarray,
        output_shape: List[int],
        output_strides: List[int],
        input: np.ndarray,
        input_shape: List[int],
        input_strides: List[int],
        diagonal: int,
        inplace: Inplace
    ):
        super().__init__("triu")
        self.output = output
        self.output_shape = output_shape
        self.output_strides = output_strides
        self.input = input
        self.input_shape = input_shape
        self.input_strides = input_strides
        self.diagonal = diagonal
        self.inplace = inplace

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.output_shape)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.input_shape)
        test_writer.add_int64(test_writer.gguf_key("diagonal"), self.diagonal)
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

        ans = triu(
            self.output,
            self.input,
            self.diagonal,
            self.inplace
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
if __name__ == "__main__":
    test_writer = InfiniopTestWriter("triu.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, diagonal
        ((5, 6), 0),
        ((4, 5), -1),
        ((61, 71), 2),
        ((111, 222), 33),
        ((1001, 1001), -101)
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16, np.int64, np.int32, np.bool_]
    for dtype in _TENSOR_DTYPES_:
        for shape, diagonal in _TEST_CASES_:
            for inplace in _INPLACE:
                input = np.random.rand(*shape).astype(dtype)
                if inplace == Inplace.INPLACE:
                    output = input
                else:
                    output = np.empty(shape, dtype=dtype)
                test_case = TriuTestCase(
                    output=output,
                    output_shape=shape,
                    output_strides=None,
                    input=input,
                    input_shape=shape,
                    input_strides=None,
                    diagonal=diagonal,
                    inplace=inplace
                )
                test_cases.append(test_case)
    test_writer.add_tests(test_cases)
    test_writer.save()
