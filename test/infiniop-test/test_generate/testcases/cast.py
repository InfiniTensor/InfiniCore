import torch
import gguf
import numpy as np
from typing import List

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

# PyTorch dtype to InfiniOP dtype mapping (only CPU supported types)
DTYPE_MAPPING = {
    torch.float32: 13,   # INFINI_DTYPE_F32
    torch.float16: 12,   # INFINI_DTYPE_F16
    torch.int32: 5,      # INFINI_DTYPE_I32
    torch.int64: 6,      # INFINI_DTYPE_I64
    # Note: CPU implementation doesn't support I8, I16 types
}

def reference_cast(input_tensor: torch.Tensor, output_dtype: torch.dtype) -> torch.Tensor:
    """Reference implementation using PyTorch cast"""
    return input_tensor.to(output_dtype)

class CastTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        output_dtype: torch.dtype,
        shape: List[int],
        stride: List[int] | None,
    ):
        super().__init__("cast")
        self.input_tensor = input_tensor
        self.output_dtype = output_dtype
        self.shape = shape
        self.stride = stride

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Add to_type attribute
        to_type_enum = DTYPE_MAPPING.get(self.output_dtype)
        if to_type_enum is None:
            raise ValueError(f"Unsupported target dtype: {self.output_dtype}")
        test_writer.add_array(test_writer.gguf_key("to_type"), [to_type_enum])
        
        # Add input shape and strides
        test_writer.add_array(test_writer.gguf_key("input.shape"), self.shape)
        strides = self.stride if self.stride is not None else contiguous_gguf_strides(self.shape)
        test_writer.add_array(test_writer.gguf_key("input.strides"), gguf_strides(*strides))
        
        # Add output shape and strides (same as input)
        test_writer.add_array(test_writer.gguf_key("output.shape"), self.shape)
        test_writer.add_array(test_writer.gguf_key("output.strides"), gguf_strides(*contiguous_gguf_strides(self.shape)))
        
        # Handle input tensor
        input_numpy = self.input_tensor.numpy()
        input_ggml_dtype = np_dtype_to_ggml(input_numpy.dtype)
        
        # Add input tensor
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=input_ggml_dtype,
        )
        
        # Create empty output tensor with target dtype
        output_tensor = torch.empty(self.shape, dtype=self.output_dtype)
        output_numpy = output_tensor.numpy()
        output_ggml_dtype = np_dtype_to_ggml(output_numpy.dtype)
        
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            output_numpy,
            raw_dtype=output_ggml_dtype,
        )
        
        # Generate expected result
        expected_output = reference_cast(self.input_tensor, self.output_dtype)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            expected_output.double().numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

if __name__ == "__main__":
    test_writer = InfiniopTestWriter("cast.gguf")
    test_cases: List[CastTestCase] = []

    _TEST_SHAPES_ = [
        (3, 3),
        (32, 512),
        (4, 4, 4),
        (16, 32, 512),
        (1024,),
        (2, 3, 4, 5),
    ]

    _TEST_STRIDES_ = [
        None,  # Contiguous only
    ]

    # Define type conversion test matrix (CPU supported types only)
    _TYPE_CONVERSIONS_: List[tuple[torch.dtype, torch.dtype]] = [
        # Integer to integer conversions
        (torch.int32, torch.int64),
        (torch.int64, torch.int32),
        
        # Float to float conversions
        (torch.float16, torch.float32),
        (torch.float32, torch.float16),
        
        # Integer to float conversions
        (torch.int32, torch.float16),
        (torch.int32, torch.float32),
        (torch.int64, torch.float16),
        (torch.int64, torch.float32),
        
        # Float to integer conversions
        (torch.float16, torch.int32),
        (torch.float16, torch.int64),
        (torch.float32, torch.int32),
        (torch.float32, torch.int64),
    ]

    for input_dtype, output_dtype in _TYPE_CONVERSIONS_:
        # Skip unsupported types
        if input_dtype not in DTYPE_MAPPING or output_dtype not in DTYPE_MAPPING:
            continue
            
        for i, shape in enumerate(_TEST_SHAPES_):
            # Use contiguous stride only
            stride = None
            
            # Generate appropriate test data based on input type
            if input_dtype in [torch.int32, torch.int64]:
                # Integer data: use small range to avoid overflow
                input_data = torch.randint(-100, 100, shape, dtype=input_dtype)
            else:
                # Float data: use normal distribution
                input_data = torch.randn(shape, dtype=torch.float32) * 2.0
                input_data = input_data.to(input_dtype)
            
            test_case = CastTestCase(
                input_data,
                output_dtype,
                list(shape),
                list(stride) if stride is not None else None,
            )
            test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
    print(f"Generated {len(test_cases)} test cases for Cast operator")