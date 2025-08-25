import torch
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_sigmoid_backward(grad_output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """Reference implementation of sigmoid backward"""
    sigmoid_input = torch.sigmoid(input)
    return grad_output * sigmoid_input * (1 - sigmoid_input)

class SigmoidBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        grad_output: torch.Tensor,
        input: torch.Tensor,
        shape: List[int] | None,
        stride: List[int] | None,
    ):
        super().__init__("sigmoid_backward")
        self.grad_output = grad_output
        self.input = input
        self.shape = shape
        self.stride = stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Add shapes
        test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.shape)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape)
        test_writer.add_array(test_writer.gguf_key("grad_input.shape"), self.shape)
        
        # Add strides
        strides = self.stride if self.stride is not None else contiguous_gguf_strides(self.shape)
        test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*strides))
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides))
        test_writer.add_array(test_writer.gguf_key("grad_input.strides"), gguf_strides(*strides))
        
        # Handle data type conversion
        if self.grad_output.dtype == torch.bfloat16:
            grad_output_numpy = self.grad_output.view(torch.uint16).numpy()
            input_numpy = self.input.view(torch.uint16).numpy()
            ggml_dtype = gguf.GGMLQuantizationType.BF16
        else:
            grad_output_numpy = self.grad_output.numpy()
            input_numpy = self.input.numpy()
            ggml_dtype = np_dtype_to_ggml(grad_output_numpy.dtype)
        
        # Add input tensors
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"),
            grad_output_numpy,
            raw_dtype=ggml_dtype,
        )
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=ggml_dtype,
        )
        
        # Create empty grad_input tensor
        import numpy as np
        grad_input_numpy = np.empty(self.shape, dtype=grad_output_numpy.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"),
            grad_input_numpy,
            raw_dtype=ggml_dtype,
        )
        
        # Generate expected answer
        ans = reference_sigmoid_backward(self.grad_output.double(), self.input.double())
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("sigmoid_backward.gguf")
    test_cases: List[SigmoidBackwardTestCase] = []

    _TEST_CASES_ = [
        ((3, 3), None),
        ((32, 512), None),
        ((32, 512), (1024, 1)),
        ((4, 4, 4), None),
        ((16, 32, 512), None),
        ((16, 20, 512), (20480, 512, 1)),
        ((1024,), None),
        ((1024,), (2,)),
        ((2, 3, 4, 5), None),
    ]

    _TENSOR_DTYPES_ = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in _TENSOR_DTYPES_:
        for shape, stride in _TEST_CASES_:
            # Generate random input data
            grad_output = torch.randn(shape, dtype=dtype)
            input = torch.randn(shape, dtype=dtype)
            
            # Apply stride if specified
            if stride is not None:
                # Create larger tensor first to accommodate the stride
                total_size = max(shape[i] * stride[i] for i in range(len(shape)))
                grad_output_large = torch.randn(total_size, dtype=dtype)
                input_large = torch.randn(total_size, dtype=dtype)
                grad_output = grad_output_large.as_strided(shape, stride)
                input = input_large.as_strided(shape, stride)
            
            test_case = SigmoidBackwardTestCase(grad_output, input, shape, stride)
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()