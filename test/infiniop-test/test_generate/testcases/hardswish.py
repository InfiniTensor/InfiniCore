import torch
import gguf
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_hardswish(input: torch.Tensor) -> torch.Tensor:
    """
    Reference implementation of HardSwish activation function.
    HardSwish(x) = x * ReLU6(x + 3) / 6
    where ReLU6(x) = min(max(x, 0), 6)
    """
    x_plus_3 = input + 3.0
    relu6_result = torch.clamp(x_plus_3, min=0.0, max=6.0)
    return input * relu6_result / 6.0

class HardSwishTestCase(InfiniopTestCase):
    def __init__(
        self,
        input: torch.Tensor,
        shape: List[int] | None,
        stride: List[int] | None,
    ):
        super().__init__("hardswish")
        self.input = input
        self.shape = shape
        self.stride = stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape)
        strides = self.stride if self.stride is not None else contiguous_gguf_strides(self.shape)    
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides))
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape)
        test_writer.add_array(test_writer.gguf_key("output.strides"), gguf_strides(*strides))
        if self.input.dtype == torch.bfloat16:
            input_numpy = self.input.view(torch.uint16).numpy()
            ggml_dtype = gguf.GGMLQuantizationType.BF16
        else:
            input_numpy = self.input.numpy()
            ggml_dtype = np_dtype_to_ggml(input_numpy.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=ggml_dtype,
        )
        # Create empty output tensor with same shape as input
        import numpy as np
        output_numpy = np.empty(self.shape, dtype=input_numpy.dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            output_numpy,
            raw_dtype=ggml_dtype,
        )
        # Generate expected answer
        ans = reference_hardswish(self.input.double())
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("hardswish.gguf")
    test_cases: List[HardSwishTestCase] = []

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
            # Generate random input data in range [-6, 6] to cover all HardSwish behavior regions
            # HardSwish has different behaviors:
            # x < -3: output = 0
            # -3 <= x <= 3: output = x * (x + 3) / 6
            # x > 3: output = x
            input_data = torch.randn(shape, dtype=torch.float32) * 3.0  # Range roughly [-9, 9]
            input_data = input_data.to(dtype)
            
            test_case = HardSwishTestCase(input_data, list(shape), stride)
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()