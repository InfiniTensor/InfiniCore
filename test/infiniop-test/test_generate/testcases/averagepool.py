from ast import List
import numpy as np
import gguf
import torch
import torch.nn as nn
from typing import List, Union, Tuple

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides

# Based on PyTorch docs, this script is implemented.


def averagepool1d(
    input_tensor: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """1D Average Pooling using PyTorch with double precision"""
    pool = nn.AvgPool1d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )
    if divisor_override is not None:
        # divisor_override must be None for AvgPool1d in PyTorch
        pass
    result = pool(input_tensor.double())
    return result


def averagepool2d(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """2D Average Pooling using PyTorch with double precision"""
    pool = nn.AvgPool2d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    result = pool(input_tensor.double())
    return result


def averagepool3d(
    input_tensor: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """3D Average Pooling using PyTorch with double precision"""
    pool = nn.AvgPool3d(
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )
    result = pool(input_tensor.double())
    return result


def random_tensor(shape, dtype):
    """Generate random tensor using torch"""
    tensor = torch.randn(shape, dtype=dtype).clamp(-1, 1)
    return tensor


class AveragePoolTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_tensor: torch.Tensor,
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = None,
        padding: Union[int, Tuple] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int = None,
        pool_dim: int = 2,  # 1, 2, or 3
    ):
        super().__init__("averagepool")
        self.input_tensor = input_tensor
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
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

        # Add parameters
        # Adding stride only if it is not None
        # Though stride maybe not in gguf, they are needed for PyTorch to compute the expected output correctly
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
            ans = averagepool1d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        elif self.pool_dim == 2:
            ans = averagepool2d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        elif self.pool_dim == 3:
            ans = averagepool3d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        else:
            raise ValueError(f"Unsupported pool dimension: {self.pool_dim}")

        # Store output in double precision
        test_writer.add_tensor(
            test_writer.gguf_key("output"),
            ans.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("averagepool.gguf")

    # Data types to test
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    test_cases = []

    # Generate comprehensive test cases for each data type and dimension
    for dtype in dtypes:

        # ============ 1D Average Pooling Tests ============
        # Basic cases
        test_cases.extend(
            [
                AveragePoolTestCase(
                    random_tensor((4, 8, 128), dtype),
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    pool_dim=1,
                ),
                AveragePoolTestCase(
                    random_tensor((2, 16, 256), dtype),
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    pool_dim=1,
                ),
                AveragePoolTestCase(
                    random_tensor((8, 4, 64), dtype),
                    kernel_size=7,
                    stride=3,
                    padding=1,
                    pool_dim=1,
                ),
            ]
        )

        # ceil_mode variations
        test_cases.extend(
            [
                AveragePoolTestCase(
                    random_tensor((1, 3, 99), dtype),
                    kernel_size=4,
                    stride=3,
                    padding=1,
                    ceil_mode=True,
                    pool_dim=1,
                ),
                AveragePoolTestCase(
                    random_tensor((3, 2, 77), dtype),
                    kernel_size=6,
                    stride=4,
                    padding=0,
                    ceil_mode=True,
                    pool_dim=1,
                ),
            ]
        )

        # ============ 2D Average Pooling Tests ============
        # Basic cases with square kernels
        test_cases.extend(
            [
                AveragePoolTestCase(
                    random_tensor((2, 3, 64, 64), dtype),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool_dim=2,
                ),
                AveragePoolTestCase(
                    random_tensor((4, 16, 128, 128), dtype),
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    pool_dim=2,
                ),
                AveragePoolTestCase(
                    random_tensor((1, 8, 96, 96), dtype),
                    kernel_size=7,
                    stride=3,
                    padding=0,
                    pool_dim=2,
                ),
            ]
        )

        # Rectangular kernels
        test_cases.extend(
            [
                AveragePoolTestCase(
                    random_tensor((2, 4, 80, 120), dtype),
                    kernel_size=(3, 5),
                    stride=(1, 2),
                    padding=(1, 2),
                    pool_dim=2,
                ),
                AveragePoolTestCase(
                    random_tensor((1, 6, 72, 48), dtype),
                    kernel_size=(7, 3),
                    stride=(2, 1),
                    padding=(3, 1),
                    pool_dim=2,
                ),
                AveragePoolTestCase(
                    random_tensor((3, 2, 56, 84), dtype),
                    kernel_size=(2, 4),
                    stride=(2, 3),
                    padding=(0, 2),
                    pool_dim=2,
                ),
            ]
        )

        # ceil_mode variations
        test_cases.extend(
            [
                AveragePoolTestCase(
                    random_tensor((1, 1, 33, 33), dtype),
                    kernel_size=4,
                    stride=3,
                    padding=1,
                    ceil_mode=True,
                    pool_dim=2,
                ),
                AveragePoolTestCase(
                    random_tensor((2, 5, 77, 89), dtype),
                    kernel_size=(5, 3),
                    stride=(4, 2),
                    padding=(2, 1),
                    ceil_mode=True,
                    pool_dim=2,
                ),
            ]
        )

        # ============ 3D Average Pooling Tests ============
        # Basic cubic kernels
        test_cases.extend(
            [
                AveragePoolTestCase(
                    random_tensor((1, 2, 32, 32, 32), dtype),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool_dim=3,
                ),
                AveragePoolTestCase(
                    random_tensor((2, 4, 48, 48, 48), dtype),
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    pool_dim=3,
                ),
                AveragePoolTestCase(
                    random_tensor((1, 1, 64, 64, 64), dtype),
                    kernel_size=7,
                    stride=3,
                    padding=0,
                    pool_dim=3,
                ),
            ]
        )

        # Non-cubic kernels
        test_cases.extend(
            [
                AveragePoolTestCase(
                    random_tensor((1, 3, 24, 36, 48), dtype),
                    kernel_size=(2, 3, 4),
                    stride=(1, 2, 2),
                    padding=(0, 1, 2),
                    pool_dim=3,
                ),
                AveragePoolTestCase(
                    random_tensor((2, 2, 40, 32, 56), dtype),
                    kernel_size=(5, 3, 7),
                    stride=(2, 1, 3),
                    padding=(2, 1, 3),
                    pool_dim=3,
                ),
                AveragePoolTestCase(
                    random_tensor((1, 1, 28, 44, 36), dtype),
                    kernel_size=(3, 5, 2),
                    stride=(2, 3, 1),
                    padding=(1, 2, 1),
                    pool_dim=3,
                ),
            ]
        )

        # ceil_mode variations
        test_cases.extend(
            [
                AveragePoolTestCase(
                    random_tensor((1, 1, 27, 27, 27), dtype),
                    kernel_size=4,
                    stride=3,
                    padding=1,
                    ceil_mode=True,
                    pool_dim=3,
                ),
                AveragePoolTestCase(
                    random_tensor((2, 2, 33, 45, 39), dtype),
                    kernel_size=(5, 3, 4),
                    stride=(3, 2, 3),
                    padding=(2, 1, 1),
                    ceil_mode=True,
                    pool_dim=3,
                ),
            ]
        )

    # Add some edge cases
    edge_cases = [
        # Very large kernels
        AveragePoolTestCase(
            random_tensor((1, 2, 64), torch.float32),
            kernel_size=32,
            stride=16,
            padding=16,
            pool_dim=1,
        ),
        AveragePoolTestCase(
            random_tensor((1, 1, 64, 64), torch.float16),
            kernel_size=32,
            stride=16,
            padding=16,
            pool_dim=2,
        ),
        # Kernel size equals input size
        AveragePoolTestCase(
            random_tensor((1, 2, 16, 16), torch.bfloat16),
            kernel_size=16,
            stride=1,
            padding=8,
            pool_dim=2,
        ),
        # Large stride
        AveragePoolTestCase(
            random_tensor((2, 3, 100, 100), torch.float32),
            kernel_size=5,
            stride=10,
            padding=2,
            pool_dim=2,
        ),
        # Complex 3D case
        AveragePoolTestCase(
            random_tensor((1, 1, 16, 32, 48), torch.float16),
            kernel_size=(8, 4, 6),
            stride=(4, 8, 12),
            padding=(4, 2, 3),
            pool_dim=3,
        ),
    ]

    test_cases.extend(edge_cases)

    print(f"Generated {len(test_cases)} test cases")
    print(f"Data types: {len(dtypes)} types")
    print(f"Pool dimensions: 1D, 2D, 3D")
    print(
        f"Note: AveragePool test uses default parameters (count_include_pad=True, divisor_override=None)"
    )

    test_writer.add_tests(test_cases)
    test_writer.save()
