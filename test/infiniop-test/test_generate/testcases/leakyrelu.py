import torch
import gguf
from typing import List
from ml_dtypes import bfloat16
from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_leakyrelu(input: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    return torch.where(input >= 0, input, input * negative_slope)

class LeakyReLUTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        output: torch.Tensor,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
        negative_slope: float,
    ):
        super().__init__("leakyrelu")
        self.input = input
        self.shape_input = shape_input
        self.stride_input = stride_input
        self.output = output
        self.shape_output = shape_output
        self.stride_output = stride_output
        self.negative_slope = negative_slope

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        if self.shape_input is not None:
            test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape_input)
        if self.shape_output is not None:
            test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape_output)

        strides_input = self.stride_input if self.stride_input is not None else contiguous_gguf_strides(self.shape_input)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides_input))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output))
        )

        test_writer.add_float32(test_writer.gguf_key("negative_slope"), self.negative_slope)
        
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

        if self.output.dtype == torch.bfloat16:
            bits_out = self.output.view(torch.uint16).cpu().numpy()
            arr_out = bits_out.view(bfloat16)
            out_raw = np_dtype_to_ggml(bfloat16)
        else:
            arr_out = self.output.cpu().numpy()
            out_raw = np_dtype_to_ggml(arr_out.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            arr_out,
            raw_dtype=out_raw,
        )

        ans = reference_leakyrelu(self.input.double(), self.negative_slope)

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
        # shape, input_stride, output_stride, negative_slope
        ((3, 3), None, None, 1.0),
        ((32, 512), None, None, 0.1),
        ((32, 512), (1024, 1), None, 0.1),
        ((32, 512), (1024, 1), (1024, 1), 0.1),
        ((4, 4, 4), None, None, 0.5),
        ((16, 32, 512), None, None, 0.5),
        ((16, 20, 512), (20480, 512, 1), None, 0.5),
        ((16, 20, 512), (20480, 512, 1), (20480, 512, 1), 0.5),
        ((1024,), None, None, 0.7),
        ((1024,), (2,), None, 0.7),
        ((1024,), (2,), (2,), 0.7),
        ((2, 3, 4, 5), None, None, 0.7),
    ]

    _TENSOR_DTYPES_ = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in _TENSOR_DTYPES_:
        suffix = _dtype_suffix(dtype)
        filename = f"leakyrelu_{suffix}.gguf"
        test_writer = InfiniopTestWriter(filename)
        test_cases: List[LeakyReLUTestCase] = []
        for shape, stride_input, stride_output, negative_slope in _TEST_CASES_:
            input_tensor = torch.randn(*shape, dtype=dtype) * 1.5
            output_tensor = torch.empty_like(input_tensor)
            test_case = LeakyReLUTestCase(
                input=input_tensor,
                shape_input=list(shape),
                stride_input=list(stride_input) if stride_input is not None else None,
                output=output_tensor,
                shape_output=list(shape),
                stride_output=list(stride_output) if stride_output is not None else None,
                negative_slope=negative_slope,
            )
            test_cases.append(test_case)

        test_writer.add_tests(test_cases)
        test_writer.save()