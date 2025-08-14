import torch
import gguf
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_sigmoid_backward(input: torch.Tensor, grad_output: torch.Tensor) -> torch.Tensor:
    sigmoid_input = torch.sigmoid(input)
    return grad_output * sigmoid_input * (1 - sigmoid_input)

class SigmoidBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        grad_output: torch.Tensor,
        shape_grad_output: List[int] | None,
        stride_grad_output: List[int] | None,
        grad_input: torch.Tensor,
        shape_grad_input: List[int] | None,
        stride_grad_input: List[int] | None,
    ):
        super().__init__("sigmoid_backward")
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
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_grad_output is not None:
            test_writer.add_array(test_writer.gguf_key("grad_output.shape"), self.shape_grad_output)
        if self.shape_grad_input is not None:
            test_writer.add_array(test_writer.gguf_key("grad_input.shape"), self.shape_grad_input)

        strides_input = self.stride_input if self.stride_input is not None else contiguous_gguf_strides(self.shape_input)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides_input))
        
        strides_grad_output = self.stride_grad_output if self.stride_grad_output is not None else contiguous_gguf_strides(self.shape_grad_output)
        test_writer.add_array(test_writer.gguf_key("grad_output.strides"), gguf_strides(*strides_grad_output))
        
        strides_grad_input = self.stride_grad_input if self.stride_grad_input is not None else contiguous_gguf_strides(self.shape_grad_input)
        test_writer.add_array(test_writer.gguf_key("grad_input.strides"), gguf_strides(*strides_grad_input))

        if self.input.dtype == torch.bfloat16:
            bits = self.input.view(torch.uint16).cpu().numpy()
            arr_in = bits.view(bfloat16)
            raw_dtype = np_dtype_to_ggml(bfloat16)
        else:
            arr_in = self.input.cpu().numpy()
            raw_dtype = np_dtype_to_ggml(arr_in.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            arr_in,
            raw_dtype=raw_dtype,
        )

        if self.grad_output.dtype == torch.bfloat16:
            bits_go = self.grad_output.view(torch.uint16).cpu().numpy()
            arr_go = bits_go.view(bfloat16)
            go_raw_dtype = np_dtype_to_ggml(bfloat16)
        else:
            arr_go = self.grad_output.cpu().numpy()
            go_raw_dtype = np_dtype_to_ggml(arr_go.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"),
            arr_go,
            raw_dtype=go_raw_dtype,
        )

        if self.grad_input.dtype == torch.bfloat16:
            bits_gi = self.grad_input.view(torch.uint16).cpu().numpy()
            arr_gi = bits_gi.view(bfloat16)
            gi_raw_dtype = np_dtype_to_ggml(bfloat16)
        else:
            arr_gi = self.grad_input.cpu().numpy()
            gi_raw_dtype = np_dtype_to_ggml(arr_gi.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"),
            arr_gi,
            raw_dtype=gi_raw_dtype,
        )

        ans = reference_sigmoid_backward(self.input.double(), self.grad_output.double())
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

def _dtype_suffix(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    s = str(dtype)
    s = s.replace("torch.", "")
    s = s.replace("numpy.", "")
    s = s.replace("<class '", "").replace("'>", "")
    return s.replace(".", "_").replace(" ", "")

if __name__ == "__main__":
    _TEST_CASES_ = [
        # (shape, input_stride, grad_output_stride, grad_input_stride)
        ((3, 3), None, None, None),
        ((32, 512), None, None, None),
        ((32, 512), (1024, 1), (1024, 1), None),
        ((32, 512), (1024, 1), (1024, 1), (1024, 1)),
        ((4, 4, 4), None, None, None),
        ((16, 32, 512), None, None, None),
        ((16, 20, 512), (20480, 512, 1), (20480, 512, 1), None),
        ((16, 20, 512), (20480, 512, 1), (20480, 512, 1), (20480, 512, 1)),
        ((1024,), None, None, None),
        ((1024,), (2,), (2,), None),
        ((1024,), (2,), (2,), (2,)),
        ((2, 3, 4, 5), None, None, None),
    ]

    _TENSOR_DTYPES_ = [torch.float16, torch.float32, torch.bfloat16]
    
    for dtype in _TENSOR_DTYPES_:
        suffix = _dtype_suffix(dtype)
        filename = f"sigmoid_backward_{suffix}.gguf"
        test_writer = InfiniopTestWriter(filename)
        test_cases: List[SigmoidBackwardTestCase] = []
        for shape, stride_input, stride_grad_output, stride_grad_input in _TEST_CASES_:
            input_tensor = torch.randn(*shape, dtype=dtype) * 2.0
            grad_output_tensor = torch.randn(*shape, dtype=dtype)
  
            grad_input_tensor = torch.empty_like(input_tensor)
            
            test_case = SigmoidBackwardTestCase(
                input=input_tensor,
                shape_input=list(shape),
                stride_input=list(stride_input) if stride_input is not None else None,
                grad_output=grad_output_tensor,
                shape_grad_output=list(shape),
                stride_grad_output=list(stride_grad_output) if stride_grad_output is not None else None,
                grad_input=grad_input_tensor,
                shape_grad_input=list(shape),
                stride_grad_input=list(stride_grad_input) if stride_grad_input is not None else None,
            )
            test_cases.append(test_case)

        test_writer.add_tests(test_cases)
        test_writer.save()
