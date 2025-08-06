import torch
import gguf
import numpy as np
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def sigmoid_backward(input: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output * input * (1 - input)

class SigmoidBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        grad_output: torch.Tensor,
        shape_grad_output: List[int] | None,
        stride_grad_output: List[int] | None,
    ):
        super().__init__("sigmoid_backward")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input        
        self.grad_output = grad_output
        self.shape_grad_output = shape_grad_output
        self.stride_grad_output = stride_grad_output

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        strides_input = self.stride_input if self.stride_input is not None else contiguous_gguf_strides(self.shape_input)    
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides_input))
        if self.input.dtype == torch.bfloat16:
            input_numpy = self.input.view(torch.uint16).numpy()
            input_dtype = gguf.GGMLQuantizationType.BF16
        else:
            input_numpy = self.input.numpy()
            input_dtype = np_dtype_to_ggml(input_numpy.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=input_dtype,
        )

        test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.shape_grad_output)
        strides_grad_output = self.stride_grad_output if self.stride_grad_output is not None else contiguous_gguf_strides(self.shape_grad_output)    
        test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*strides_grad_output))
        if self.grad_output.dtype == torch.bfloat16:
            grad_output_numpy = self.grad_output.view(torch.uint16).numpy()
            grad_output_dtype = gguf.GGMLQuantizationType.BF16
        else:
            grad_output_numpy = self.grad_output.numpy()
            grad_output_dtype = np_dtype_to_ggml(grad_output_numpy.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"),
            grad_output_numpy,
            raw_dtype=grad_output_dtype,
        )

        grad_input = sigmoid_backward(self.input, self.grad_output).double()
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"),
            grad_input.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("sigmoid_backward.gguf")
    test_cases: List[SigmoidBackwardTestCase] = []

    _TEST_CASES_ = [
        ((1,), None),
        ((8,), (2,)),
        ((4, 4), None),
        ((4, 4), (4, 1)),
        ((4, 4), (1, 4)),
        ((8, 16), None),
        ((32, 64), None),
        ((8, 8, 8), None),
        ((8, 8, 8), (64, 8, 1)),
        ((4, 5, 6, 7), None),
        ((4, 5, 6, 7), (210, 42, 7, 1)),
        ((1024,), None),
    ]

    _TENSOR_DTYPES_ = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in _TENSOR_DTYPES_:
        for shape, stride in _TEST_CASES_:
            input = torch.rand(*shape, dtype=dtype)
            grad_output = torch.rand(*shape, dtype=dtype)

            test_case = SigmoidBackwardTestCase(
                input=input,
                shape_input=list(shape),
                stride_input=list(stride) if stride is not None else None,
                grad_output=grad_output,
                shape_grad_output=list(shape),
                stride_grad_output=list(stride) if stride is not None else None,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
