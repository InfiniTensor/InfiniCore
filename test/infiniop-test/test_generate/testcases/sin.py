import torch
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_sin(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(input)

class SinTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        shape_input: List[int] | None,
        stride_input: List[int] | None,
        output: torch.Tensor,
        shape_output: List[int] | None,
        stride_output: List[int] | None,
    ):
        super().__init__("sin")
        self.input = input
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

        strides_input = self.stride_input if self.stride_input is not None else contiguous_gguf_strides(self.shape_input)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides_input))
        test_writer.add_array(
            test_writer.gguf_key("output.strides"),
            gguf_strides(*self.stride_output if self.stride_output is not None else contiguous_gguf_strides(self.shape_output))
        )

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

        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            self.output.numpy()
            if self.output.dtype != torch.bfloat16
            else self.output.view(torch.uint16).numpy(),
            raw_dtype=input_dtype,
        )

        ans = reference_sin(self.input.double())

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
        # shape, input_stride, output_stride
        ((3, 3), None, None),
        ((32, 512), None, None),
        ((32, 512), (1024, 1), None),
        ((32, 512), (1024, 1), (1024, 1)),
        ((4, 4, 4), None, None),
        ((16, 32, 512), None, None),
        ((16, 20, 512), (20480, 512, 1), None),
        ((16, 20, 512), (20480, 512, 1), (20480, 512, 1)),
        ((1024,), None, None),
        ((1024,), (2,), None),
        ((1024,), (2,), (2,)),
        ((2, 3, 4, 5), None, None),
    ]

    _TENSOR_DTYPES_ = [torch.float16, torch.float32, torch.bfloat16]

    for dtype in _TENSOR_DTYPES_:
        suffix = _dtype_suffix(dtype)
        filename = f"sin_{suffix}.gguf"
        test_writer = InfiniopTestWriter(filename)
        test_cases: List[SinTestCase] = []
        for shape, stride_input, stride_output in _TEST_CASES_:
            input_tensor = (torch.rand(*shape, dtype=dtype) * 4 - 2) * torch.pi
            output_tensor = torch.empty_like(input_tensor)

            test_case = SinTestCase(
                input=input_tensor,
                shape_input=list(shape),
                stride_input=list(stride_input) if stride_input is not None else None,
                output=output_tensor,
                shape_output=list(shape),
                stride_output=list(stride_output) if stride_output is not None else None,
            )
            test_cases.append(test_case)

        test_writer.add_tests(test_cases)
        test_writer.save()