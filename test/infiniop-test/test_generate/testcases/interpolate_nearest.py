import numpy as np
import gguf
import torch
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def interpolate_nearest_1d(
    input_tensor: torch.Tensor,
    size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
):
    """1D Interpolate Nearest using PyTorch with double precision"""
    return F.interpolate(
        input_tensor.double(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def interpolate_nearest_2d(
    input_tensor: torch.Tensor,
    size: Optional[Union[int, Tuple[int, int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
):
    """2D Interpolate Nearest using PyTorch with double precision"""
    return F.interpolate(
        input_tensor.double(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def interpolate_nearest_3d(
    input_tensor: torch.Tensor,
    size: Optional[Union[int, Tuple[int, int, int]]] = None,
    scale_factor: Optional[Union[float, Tuple[float, float, float]]] = None,
    mode: str = "nearest",
    align_corners: Optional[bool] = None,
    recompute_scale_factor: Optional[bool] = None,
    antialias: bool = False,
):
    """3D Interpolate Nearest using PyTorch with double precision"""
    return F.interpolate(
        input_tensor.double(),
        size=size,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
        recompute_scale_factor=recompute_scale_factor,
        antialias=antialias,
    )


def random_tensor(shape, dtype):
    """Generate random tensor using torch"""
    if dtype == torch.int8:
        # For int8, generate random integers in the range [-128, 127]
        tensor = torch.randint(-128, 128, shape, dtype=dtype)
    else:
        tensor = torch.randn(shape, dtype=dtype).clamp(-1, 1)
    return tensor


class InterpolateNearestTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        size: Optional[Union[int, Tuple]] = None,
        scale_factor: Optional[Union[float, Tuple]] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        antialias: bool = False,
        interp_dim: int = 2,  # 1, 2, or 3
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__("interpolate_nearest")
        self.input_tensor = random_tensor(input_size, dtype)
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias
        self.interp_dim = interp_dim

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # Handle different data types for input - keep original data type
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

        # Compute expected output using double precision
        if self.input_tensor.dtype == torch.int8:
            # For int8, convert to float first then to double
            input_float = self.input_tensor.float()
            if self.interp_dim == 1:
                ans_float = interpolate_nearest_1d(
                    input_float,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            elif self.interp_dim == 2:
                ans_float = interpolate_nearest_2d(
                    input_float,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            elif self.interp_dim == 3:
                ans_float = interpolate_nearest_3d(
                    input_float,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            else:
                raise ValueError(
                    f"Unsupported interpolation dimension: {self.interp_dim}"
                )

            # Convert back to int8 then to double for output
            ans = ans_float.round().clamp(-128, 127).to(torch.int8).double()
        else:
            # Use double precision computation directly
            if self.interp_dim == 1:
                ans = interpolate_nearest_1d(
                    self.input_tensor,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            elif self.interp_dim == 2:
                ans = interpolate_nearest_2d(
                    self.input_tensor,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            elif self.interp_dim == 3:
                ans = interpolate_nearest_3d(
                    self.input_tensor,
                    self.size,
                    self.scale_factor,
                    self.mode,
                    self.align_corners,
                    self.recompute_scale_factor,
                    self.antialias,
                )
            else:
                raise ValueError(
                    f"Unsupported interpolation dimension: {self.interp_dim}"
                )

        # Add expected output tensor in double precision
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

    # ==============================================================================
    #  Configuration
    # ==============================================================================
    # These are not meant to be imported from other modules
    # Format: (input_size, size, interp_dim)
    _TEST_CASES = [
        # 1D Interpolation Tests
        ((4, 8, 32), 64, 1),
        ((2, 16, 128), 256, 1),
        ((1, 4, 100), 50, 1),
        ((2, 6, 48), 96, 1),
        ((1, 3, 64), 32, 1),
        ((3, 2, 80), 120, 1),
        
        # 2D Interpolation Tests
        ((2, 3, 32, 32), (64, 64), 2),
        ((4, 16, 128, 128), (256, 256), 2),
        ((1, 8, 96, 96), (48, 48), 2),
        ((2, 4, 40, 60), (80, 120), 2),
        ((1, 6, 72, 48), (36, 96), 2),
        ((3, 2, 56, 84), (28, 42), 2),
        
        # Non-uniform size cases
        ((1, 3, 32, 48), (64, 72), 2),
        ((2, 5, 60, 40), (45, 50), 2),
        ((1, 1, 80, 60), (40, 120), 2),
        
        # 3D Interpolation Tests
        ((1, 2, 16, 16, 16), (32, 32, 32), 3),
        ((2, 4, 24, 24, 24), (48, 48, 48), 3),
        ((1, 1, 40, 40, 40), (20, 20, 20), 3),
        ((1, 3, 12, 18, 24), (24, 36, 48), 3),
        ((2, 2, 20, 16, 28), (10, 32, 14), 3),
        ((1, 1, 32, 24, 16), (16, 48, 32), 3),
        ((1, 2, 18, 24, 30), (27, 18, 24), 3),
    ]

    # Add edge cases for specific data types
    if dtype == torch.float32:
        _TEST_CASES.extend([
            ((1, 1, 4), 16, 1),
            ((1, 1, 4, 4), (16, 16), 2),
            ((1, 1, 4, 4, 4), (8, 8, 8), 3),
            ((1, 1, 16, 16), (48, 48), 2),
            ((1, 3, 1), 10, 1),
        ])

    for input_size, size, interp_dim in _TEST_CASES:
        test_case = InterpolateNearestTestCase(
            input_size=input_size,
            size=size,
            mode="nearest",
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False,
            interp_dim=interp_dim,
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
        torch.int8,
    ]
    
    dtype_filename_map = {
        torch.float32: "interpolate_nearest_float32.gguf",
        torch.float16: "interpolate_nearest_float16.gguf",
        torch.bfloat16: "interpolate_nearest_bfloat16.gguf",
        torch.int8: "interpolate_nearest_int8.gguf",
    }

    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        print(f"Generating {filename} for dtype {dtype}")
        gen_gguf(dtype, filename)
        
    print(f"Generated GGUF files for {len(_TENSOR_DTYPES_)} data types")
    print("Interpolation dimensions: 1D, 2D, 3D")
    print("Mode: nearest only, all tensors use contiguous memory")
 