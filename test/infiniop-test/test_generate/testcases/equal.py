import torch
import gguf
import numpy as np
from typing import List
from ml_dtypes import bfloat16

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides, contiguous_gguf_strides

def reference_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Reference implementation using PyTorch equal"""
    return torch.equal(a, b)

class EqualTestCase(InfiniopTestCase):
    def __init__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        shape: List[int],
        stride_a: List[int] | None,
        stride_b: List[int] | None,
    ):
        super().__init__("equal")
        self.a = a
        self.b = b
        self.shape = shape
        self.stride_a = stride_a
        self.stride_b = stride_b

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        
        # Add input shapes and strides
        test_writer.add_array(test_writer.gguf_key("a.shape"), self.shape)
        strides_a = self.stride_a if self.stride_a is not None else contiguous_gguf_strides(self.shape)
        if strides_a:
            test_writer.add_array(test_writer.gguf_key("a.strides"), gguf_strides(*strides_a))
        else:
            test_writer.add_array(test_writer.gguf_key("a.strides"), [])
        
        test_writer.add_array(test_writer.gguf_key("b.shape"), self.shape)
        strides_b = self.stride_b if self.stride_b is not None else contiguous_gguf_strides(self.shape)
        if strides_b:
            test_writer.add_array(test_writer.gguf_key("b.strides"), gguf_strides(*strides_b))
        else:
            test_writer.add_array(test_writer.gguf_key("b.strides"), [])
        
        # Add output shape and strides (scalar: shape [1])
        test_writer.add_array(test_writer.gguf_key("c.shape"), [1])
        test_writer.add_array(test_writer.gguf_key("c.strides"), [1])
        
        # Handle input tensors
        # Convert bfloat16 tensors to numpy using ml_dtypes
        if self.a.dtype == torch.bfloat16:
            a_numpy = self.a.detach().cpu().to(torch.float32).numpy().astype(bfloat16)
        else:
            a_numpy = self.a.numpy()
        a_ggml_dtype = np_dtype_to_ggml(a_numpy.dtype)
        
        if self.b.dtype == torch.bfloat16:
            b_numpy = self.b.detach().cpu().to(torch.float32).numpy().astype(bfloat16)
        else:
            b_numpy = self.b.numpy()
        b_ggml_dtype = np_dtype_to_ggml(b_numpy.dtype)
        
        # Add input tensors
        test_writer.add_tensor(
            test_writer.gguf_key("a"),
            a_numpy,
            raw_dtype=a_ggml_dtype,
        )
        
        test_writer.add_tensor(
            test_writer.gguf_key("b"),
            b_numpy,
            raw_dtype=b_ggml_dtype,
        )
        
        # Create output tensor with shape (1,) for scalar result
        c_tensor = torch.empty((1,), dtype=torch.bool)
        c_numpy = c_tensor.numpy()
        
        test_writer.add_tensor(
            test_writer.gguf_key("c"),
            c_numpy,
            raw_dtype=np_dtype_to_ggml(c_numpy.dtype),
        )
        
        # Generate expected result
        expected_result = reference_equal(self.a, self.b)
        # Convert scalar bool to numpy array with shape (1,)
        ans_array = np.array([expected_result], dtype=np.bool_)
        test_writer.add_tensor(
            test_writer.gguf_key("ans"),
            ans_array,
            raw_dtype=np_dtype_to_ggml(ans_array.dtype),
        )

if __name__ == "__main__":
    # Set random seed for reproducible test cases
    torch.manual_seed(42)
    np.random.seed(42)
    
    test_writer = InfiniopTestWriter("equal.gguf")
    test_cases: List[EqualTestCase] = []

    _TEST_SHAPES_ = [
        (3, 3),
        (32, 512),
        (4, 4, 4),
        (16, 32, 512),
        (2, 3, 4, 5),
        (1024,),  # Add 1D test case that user mentioned
    ]

    _TEST_STRIDES_ = [
        None,  # Contiguous only
    ]

    # Define supported dtypes (CPU supported types)
    _TENSOR_DTYPES_ = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
        torch.int32,
        torch.int64,
    ]

    for dtype in _TENSOR_DTYPES_:
        for i, shape in enumerate(_TEST_SHAPES_):
            for stride in _TEST_STRIDES_:
                # Create test data
                if dtype in [torch.int32, torch.int64]:
                    # Integer data
                    a_data = torch.randint(-100, 100, shape, dtype=dtype)
                    # Create identical tensor for some cases, different for others
                    if i % 2 == 0:
                        b_data = a_data.clone()  # Should return True
                    else:
                        b_data = torch.randint(-100, 100, shape, dtype=dtype)  # Likely False
                else:
                    # Float data
                    a_data = torch.randn(shape, dtype=torch.float32) * 2.0
                    a_data = a_data.to(dtype)
                    if i % 2 == 0:
                        b_data = a_data.clone()  # Should return True
                    else:
                        b_data = torch.randn(shape, dtype=torch.float32) * 2.0
                        b_data = b_data.to(dtype)  # Likely False
                
                test_case = EqualTestCase(
                    a_data,
                    b_data,
                    list(shape),
                    list(stride) if stride is not None else None,
                    list(stride) if stride is not None else None,
                )
                test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()
    print(f"Generated {len(test_cases)} test cases for Equal operator")