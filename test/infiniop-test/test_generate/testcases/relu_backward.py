from ast import List
import numpy as np
import gguf
from typing import List
from numpy.lib.stride_tricks import as_strided

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides, process_zero_stride_tensor
import torch

def relu_backward(
    input: np.ndarray,
    grad_output: np.ndarray,
):
    torch_input = torch.from_numpy(input)
    torch_input.requires_grad_(True)
    torch_grad_output = torch.from_numpy(grad_output)
    torch_output = torch.relu(torch_input)
    torch_output.backward(torch_grad_output)

    return torch_input.grad.detach().numpy()


class ReLUBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: np.ndarray,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        grad_output: np.ndarray,
        shape_grad_output: List[int] | None,
        stride_grad_output: List[int] | None,
        grad_input: np.ndarray,
        shape_grad_input: List[int] | None,
        stride_grad_input: List[int] | None,

    ):
        super().__init__("relu_backward")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.grad_output = grad_output
        self.shape_grad_output = shape_grad_output
        self.stride_grad_output = stride_grad_output
        self.grad_input = grad_input
        self.shape_grad_input = shape_grad_input
        self.stride_grad_input = stride_grad_input


    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.shape_grad_output)
        test_writer.add_array(test_writer.gguf_key("grad_input.shape"), self.shape_grad_output)
        if self.stride_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*self.stride_input))
        if self.stride_grad_output is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*self.stride_grad_output))
        test_writer.add_array(
            test_writer.gguf_key("grad_input.strides"),
            gguf_strides(*self.stride_grad_input if self.stride_grad_input is not None else contiguous_gguf_strides(self.shape_grad_input))
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input"), self.input, raw_dtype=np_dtype_to_ggml(self.input.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"), self.grad_output, raw_dtype=np_dtype_to_ggml(self.grad_output.dtype)
        )
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"), self.grad_input, raw_dtype=np_dtype_to_ggml(self.grad_input.dtype)
        )
        ans = relu_backward(
            self.input.astype(np.float64),
            self.grad_output.astype(np.float64),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("relu_backward.gguf")
    test_cases = []
    # ==============================================================================
    #  Configuration (Internal Use Only)
    # ==============================================================================
    # These are not meant to be imported from other modules
    _TEST_CASES_ = [
        # shape, a_stride, b_stride, c_stride
        ((13, 4), None, None, None),
        ((13, 4), (10, 1), (10, 1), (10, 1)),
        ((13, 4, 4), None, None, None),
        ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
        ((16, 5632), None, None, None),
        ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
        ((4, 4, 5632), None, None, None),
        ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1), (45056, 5632, 1)),
    ]
    _TENSOR_DTYPES_ = [np.float32, np.float16]
    for dtype in _TENSOR_DTYPES_:
        for shape, stride_input, stride_grad_output, stride_grad_input in _TEST_CASES_:
            input = np.random.rand(*shape).astype(dtype) * 2 - 1
            grad_output = np.random.rand(*shape).astype(dtype) * 2 - 1
            grad_input = np.empty(tuple(0 for _ in shape), dtype=dtype)
            input = process_zero_stride_tensor(input, stride_input)
            grad_output = process_zero_stride_tensor(grad_output, stride_grad_output)
            test_case = ReLUBackwardTestCase(
                input=input,
                shape_input=shape,
                stride_input=stride_input,
                grad_output=grad_output,
                shape_grad_output=shape,
                stride_grad_output=stride_grad_output,
                grad_input=grad_input,
                shape_grad_input=shape,
                stride_grad_input=stride_grad_input,
            )
            test_cases.append(test_case)
            
    test_writer.add_tests(test_cases)
    test_writer.save()
    