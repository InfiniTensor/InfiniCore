import numpy as np
import gguf
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Union, Tuple

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides

# Based on PyTorch docs, this script is implemented.


def maxpool1d(
    input_tensor: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    dilation: int = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
):
    """1D Max Pooling using PyTorch with double precision"""
    result = F.max_pool1d(
        input_tensor.double(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

    if return_indices:
        return result[0], result[1]  # Return both values and indices
    return result, None


def maxpool2d(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
):
    """2D Max Pooling using PyTorch with double precision"""
    result = F.max_pool2d(
        input_tensor.double(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

    if return_indices:
        return result[0], result[1]  # Return both values and indices
    return result, None


def maxpool3d(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    ceil_mode: bool = False,
    return_indices: bool = False,
):
    """3D Max Pooling using PyTorch with double precision"""
    result = F.max_pool3d(
        input_tensor.double(),
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        ceil_mode=ceil_mode,
        return_indices=return_indices,
    )

    if return_indices:
        return result[0], result[1]  # Return both values and indices
    return result, None


def random_tensor(shape, dtype):
    """Generate random tensor using torch"""
    tensor = torch.randn(shape, dtype=dtype).clamp(-1, 1)
    return tensor


class MaxPoolTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = None,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        ceil_mode: bool = False,
        return_indices: bool = False,
        pool_dim: int = 2,  # 1, 2, or 3
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__("maxpool")
        self.input_tensor = random_tensor(input_size, dtype)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode
        self.return_indices = return_indices
        self.pool_dim = pool_dim

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # Keep input tensor in original data type
        if self.input_tensor.dtype == torch.bfloat16:
            input_numpy = self.input_tensor.view(torch.uint16).numpy()
            ggml_dtype = gguf.GGMLQuantizationType.BF16
        else:
            input_numpy = self.input_tensor.numpy()
            ggml_dtype = np_dtype_to_ggml(input_numpy.dtype)

        # Add input tensor
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=ggml_dtype,
        )
        test_writer.add_array(
            test_writer.gguf_key("input_shape"), 
            list(self.input_tensor.shape)
        )

        # Add parameters
        if isinstance(self.kernel_size, int):
            test_writer.add_array(
                test_writer.gguf_key("kernel_size"), [self.kernel_size]
            )
        else:
            test_writer.add_array(
                test_writer.gguf_key("kernel_size"), list(self.kernel_size)
            )

        if self.stride is not None:
            if isinstance(self.stride, int):
                test_writer.add_array(test_writer.gguf_key("stride"), [self.stride])
            else:
                test_writer.add_array(test_writer.gguf_key("stride"), list(self.stride))

        if isinstance(self.padding, int):
            test_writer.add_array(test_writer.gguf_key("padding"), [self.padding])
        else:
            test_writer.add_array(test_writer.gguf_key("padding"), list(self.padding))

        test_writer.add_bool(test_writer.gguf_key("ceil_mode"), self.ceil_mode)

        # Compute expected output using double precision
        if self.pool_dim == 1:
            ans, indices = maxpool1d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )
        elif self.pool_dim == 2:
            ans, indices = maxpool2d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )
        elif self.pool_dim == 3:
            ans, indices = maxpool3d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.dilation,
                self.ceil_mode,
                self.return_indices,
            )
        else:
            raise ValueError(f"Unsupported pool dimension: {self.pool_dim}")

        # Store output in double precision
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            ans.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )
        test_writer.add_array(
            test_writer.gguf_key("output_shape"), 
            list(ans.shape)
        )


def gen_gguf(dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # Format: (input_size, kernel_size, stride, padding, dilation, ceil_mode, pool_dim)
    _TEST_CASES = [
        # ============ 1D Max Pooling Tests ============
        # Basic cases
        ((4, 8, 128), 3, 1, 1, 1, False, 1),
        ((2, 16, 256), 5, 2, 2, 1, False, 1),
        ((8, 4, 64), 7, 3, 3, 1, False, 1),
        # ceil_mode variations
        ((1, 3, 99), 4, 3, 2, 1, True, 1),
        ((3, 2, 77), 6, 4, 3, 1, True, 1),
        
        # ============ 2D Max Pooling Tests ============
        # Basic cases with square kernels
        ((2, 3, 64, 64), 3, 1, 1, 1, False, 2),
        ((4, 16, 128, 128), 5, 2, 2, 1, False, 2),
        ((1, 8, 96, 96), 7, 3, 3, 1, False, 2),
        # Rectangular kernels
        ((2, 4, 80, 120), (3, 5), (1, 2), (1, 2), 1, False, 2),
        ((1, 6, 72, 48), (7, 3), (2, 1), (3, 1), 1, False, 2),
        ((3, 2, 56, 84), (2, 4), (2, 3), (1, 2), 1, False, 2),
        # ceil_mode variations
        ((1, 1, 33, 33), 4, 3, 2, 1, True, 2),
        ((2, 5, 77, 89), (5, 3), (4, 2), (2, 1), 1, True, 2),
        
        # ============ 3D Max Pooling Tests ============
        # Basic cubic kernels
        ((1, 2, 32, 32, 32), 3, 1, 1, 1, False, 3),
        ((2, 4, 48, 48, 48), 5, 2, 2, 1, False, 3),
        ((1, 1, 64, 64, 64), 7, 3, 3, 1, False, 3),
        # Non-cubic kernels
        ((1, 3, 24, 36, 48), (2, 3, 4), (1, 2, 2), (1, 1, 2), 1, False, 3),
        ((2, 2, 40, 32, 56), (5, 3, 7), (2, 1, 3), (2, 1, 3), 1, False, 3),
        ((1, 1, 28, 44, 36), (3, 5, 2), (2, 3, 1), (1, 2, 1), 1, False, 3),
        # ceil_mode variations
        ((1, 1, 27, 27, 27), 4, 3, 2, 1, True, 3),
        ((2, 2, 33, 45, 39), (5, 3, 4), (3, 2, 3), (2, 1, 2), 1, True, 3),
    ]

    for input_size, kernel_size, stride, padding, dilation, ceil_mode, pool_dim in _TEST_CASES:
        test_case = MaxPoolTestCase(
            input_size=input_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
            return_indices=False,
            pool_dim=pool_dim,
            dtype=dtype,
        )
        test_cases.append(test_case)

    test_writer.add_tests(test_cases)
    test_writer.save()


if __name__ == "__main__":
    # Data types to test
    _TENSOR_DTYPES_ = [
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ]
    
    dtype_filename_map = {
        torch.float32: "maxpool_float32.gguf",
        torch.float16: "maxpool_float16.gguf",
        torch.bfloat16: "maxpool_bfloat16.gguf",
    }

    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        print(f"Generating {filename} for dtype {dtype}")
        gen_gguf(dtype, filename)
        
    print(f"Generated GGUF files for {len(_TENSOR_DTYPES_)} data types")
    print("Pool dimensions: 1D, 2D, 3D")
    print("Note: MaxPool test uses default parameters (return_indices=False, dilation=1)")
    print("All tensors use contiguous memory")
