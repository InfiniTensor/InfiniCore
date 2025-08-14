import gguf
import torch
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor


def reduce_max(
    input: np.ndarray,
    dim: int,
) -> np.ndarray:
    output = torch.max(torch.from_numpy(input), dim, keepdim=True).values
    return output.numpy()


class ReduceMaxTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        output: np.ndarray,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
        dim: int,
    ):
        super().__init__("reduce_max")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.dim = dim
        
    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_int64(test_writer.gguf_key("dim"), self.dim)
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
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("output"), self.output, raw_dtype=np_dtype_to_ggml(self.output.dtype)
        )
        
        ans = reduce_max(
            self.input.astype(np.float64),
            self.dim,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )
        
def gen_gguf(dtype: np.dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []
    
    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape_input, stride_input, shape_output, stride_output, dim
        ((13, 4), None, (1, 4), None, 0),
        ((13, 4), None, (13, 1), None, 1),
        ((13, 4), (10, 1), (13, 1), (10, 1), 1),
        ((13, 4, 4), None, (1, 4, 4), None, 0),
        ((13, 4, 4), None, (13, 4, 1), None, 2),
        ((16, 5632), None, (16, 1), None, 1),
        ((16, 5632), (6000, 1), (1, 5632), (6000, 1), 0),
        ((4, 4, 5632), None, (4, 4, 1), None, 2),
        ((16, 8, 4, 8), None, (1, 8, 4, 8), None, 0),
        ((16, 8, 4, 8), None, (16, 8, 1, 8), None, 2),
    ]

    for shape_input, stride_input, shape_output, stride_output, dim in _TEST_CASES_:
        input = np.random.rand(*shape_input).astype(dtype)
        output = np.empty(tuple(0 for _ in shape_output), dtype=dtype)
        input = process_zero_stride_tensor(input, stride_input)
        output = process_zero_stride_tensor(output, stride_output)
        test_case = ReduceMaxTestCase(
            input=input,
            shape_input=shape_input,
            stride_input=stride_input,
            output=output,
            shape_output=shape_output,
            stride_output=stride_output,
            dim=dim,
        )
        test_cases.append(test_case)
    
    test_writer.add_tests(test_cases)
    test_writer.save()
    
    
if __name__ == "__main__":
    _TENSOR_DTYPES_ = [
        np.float32,
        np.float16,
        bfloat16,
    ]
    dtype_filename_map = {
        np.float32: "reduce_max_f32.gguf",
        np.float16: "reduce_max_f16.gguf",
        bfloat16: "reduce_max_bf16.gguf",
    }
    
    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        gen_gguf(dtype, filename)
