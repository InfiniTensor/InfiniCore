from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def sin(
    input_tensor: np.ndarray,
):
    return np.sin(input_tensor)


class SinTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_tensor: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        output: np.ndarray,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
    ):
        super().__init__("sin")
        self.input_tensor = input_tensor
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_output is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)
        if self.stride_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input_tensor, raw_dtype=np_dtype_to_ggml(self.input_tensor.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype)
        )
        ans = sin(
            self.input_tensor.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("sin.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, input_stride, output_stride
        ((13, 4), None, None),
        ((13, 4), (10, 1), (10, 1)),
        ((13, 4), (0, 1), None),
        ((13, 4, 4), None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
        ((13, 4, 4), (4, 0, 1), None),
        ((16, 5632), None, None),
        ((16, 5632), (13312, 1), (13312, 1)),
        ((4, 4, 5632), None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_input, stride_output in _TEST_CASES_:
            input_tensor = np.random.rand(*shape).astype(dtype)
            output = np.empty(tuple(0 for _ in shape), dtype=dtype)
            input_tensor = process_zero_stride_tensor(input_tensor, stride_input)
            test_case = SinTestCase(
                input_tensor=input_tensor,
                shape_input=shape,
                stride_input=stride_input,
                output=output,
                shape_output=shape,
                stride_output=stride_output,
            )
            test_cases.append(test_case)
            
    test_writer.add_tests(test_cases)
    test_writer.save() 