import numpy as np
import gguf
import torch
import torch.nn.functional as F
from typing import List, Union, Tuple

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def averagepool_backward_1d(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """1D Average Pooling Backward using PyTorch with double precision"""
    # Convert to double precision for computation
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.avg_pool1d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
    )

    # 反向传播 - 使用提供的grad_output
    output.backward(grad_output)

    return output, input_tensor.grad


def averagepool_backward_2d(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]] = None,
    padding: Union[int, Tuple[int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """2D Average Pooling Backward using PyTorch with double precision"""
    # Convert to double precision for computation
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.avg_pool2d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    # 反向传播 - 使用提供的grad_output
    output.backward(grad_output)

    return output, input_tensor.grad


def averagepool_backward_3d(
    input_tensor: torch.Tensor,
    grad_output: torch.Tensor,
    kernel_size: Union[int, Tuple[int, int, int]],
    stride: Union[int, Tuple[int, int, int]] = None,
    padding: Union[int, Tuple[int, int, int]] = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: int = None,
):
    """3D Average Pooling Backward using PyTorch with double precision"""
    # Convert to double precision for computation (handles half precision limitation)
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.avg_pool3d(
        input_tensor,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        divisor_override=divisor_override,
    )

    # 反向传播 - 使用提供的grad_output
    output.backward(grad_output)

    return output, input_tensor.grad


def random_tensor(shape, dtype):
    """Generate random tensor using torch"""
    tensor = torch.randn(shape, dtype=dtype).clamp(-1, 1)
    return tensor


class AveragePoolBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        kernel_size: Union[int, Tuple],
        stride: Union[int, Tuple] = None,
        padding: Union[int, Tuple] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: int = None,
        pool_dim: int = 2,  # 1, 2, or 3
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__("averagepool_backward")
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override
        self.pool_dim = pool_dim

        # 随机生成输入张量
        self.input_tensor = random_tensor(input_size, dtype)

        # 先进行前向传播获取输出形状
        if self.pool_dim == 1:
            forward_output = F.avg_pool1d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
            )
        elif self.pool_dim == 2:
            forward_output = F.avg_pool2d(
                self.input_tensor,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        elif self.pool_dim == 3:
            # Handle half precision limitation for 3D pooling
            if dtype in [torch.bfloat16, torch.float16]:
                temp_input = self.input_tensor.float()
            else:
                temp_input = self.input_tensor
            forward_output = F.avg_pool3d(
                temp_input,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
            if dtype in [torch.bfloat16, torch.float16]:
                forward_output = forward_output.to(dtype)
        else:
            raise ValueError(f"Unsupported pool dimension: {self.pool_dim}")

        # 生成随机的grad_output
        self.grad_output = random_tensor(forward_output.shape, dtype)
        self.forward_output = forward_output

        # 计算反向梯度
        if self.pool_dim == 1:
            _, self.backward_output = averagepool_backward_1d(
                self.input_tensor,
                self.grad_output,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        elif self.pool_dim == 2:
            _, self.backward_output = averagepool_backward_2d(
                self.input_tensor,
                self.grad_output,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )
        elif self.pool_dim == 3:
            _, self.backward_output = averagepool_backward_3d(
                self.input_tensor,
                self.grad_output,
                self.kernel_size,
                self.stride,
                self.padding,
                self.ceil_mode,
                self.count_include_pad,
                self.divisor_override,
            )

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # 处理grad_output（反向传播的输入）
        if self.grad_output.dtype == torch.bfloat16:
            grad_output_numpy = self.grad_output.view(torch.uint16).detach().numpy()
            ggml_dtype_input = gguf.GGMLQuantizationType.BF16
        else:
            grad_output_numpy = self.grad_output.detach().numpy()
            ggml_dtype_input = np_dtype_to_ggml(grad_output_numpy.dtype)

        # Add grad_output tensor (input for backward pass)
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"),
            grad_output_numpy,
            raw_dtype=ggml_dtype_input,
        )

        # Add complete input tensor (原始输入)
        if self.input_tensor.dtype == torch.bfloat16:
            input_numpy = self.input_tensor.view(torch.uint16).detach().numpy()
            ggml_dtype_input_tensor = gguf.GGMLQuantizationType.BF16
        else:
            input_numpy = self.input_tensor.detach().numpy()
            ggml_dtype_input_tensor = np_dtype_to_ggml(input_numpy.dtype)

        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=ggml_dtype_input_tensor,
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

        # Add backward_output tensor (output of backward pass) - 使用double精度
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"),
            self.backward_output.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("averagepool_backward.gguf")

    # Data types to test
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    test_cases = []

    # Generate comprehensive test cases for each data type and dimension
    for dtype in dtypes:

        # ============ 1D Average Pooling Backward Tests ============
        # Basic cases
        test_cases.extend(
            [
                AveragePoolBackwardTestCase(
                    input_size=(4, 8, 128),
                    kernel_size=3,
                    stride=1,
                    padding=0,
                    pool_dim=1,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(2, 16, 256),
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    pool_dim=1,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(8, 4, 64),
                    kernel_size=7,
                    stride=3,
                    padding=1,
                    pool_dim=1,
                    dtype=dtype,
                ),
            ]
        )

        # ceil_mode variations
        test_cases.extend(
            [
                AveragePoolBackwardTestCase(
                    input_size=(1, 3, 99),
                    kernel_size=4,
                    stride=3,
                    padding=1,
                    ceil_mode=True,
                    pool_dim=1,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(3, 2, 77),
                    kernel_size=6,
                    stride=4,
                    padding=0,
                    ceil_mode=True,
                    pool_dim=1,
                    dtype=dtype,
                ),
            ]
        )

        # ============ 2D Average Pooling Backward Tests ============
        # Basic cases with square kernels
        test_cases.extend(
            [
                AveragePoolBackwardTestCase(
                    input_size=(2, 3, 64, 64),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool_dim=2,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(4, 16, 128, 128),
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    pool_dim=2,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(1, 8, 96, 96),
                    kernel_size=7,
                    stride=3,
                    padding=0,
                    pool_dim=2,
                    dtype=dtype,
                ),
            ]
        )

        # Rectangular kernels
        test_cases.extend(
            [
                AveragePoolBackwardTestCase(
                    input_size=(2, 4, 80, 120),
                    kernel_size=(3, 5),
                    stride=(1, 2),
                    padding=(1, 2),
                    pool_dim=2,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(1, 6, 72, 48),
                    kernel_size=(7, 3),
                    stride=(2, 1),
                    padding=(3, 1),
                    pool_dim=2,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(3, 2, 56, 84),
                    kernel_size=(2, 4),
                    stride=(2, 3),
                    padding=(0, 2),
                    pool_dim=2,
                    dtype=dtype,
                ),
            ]
        )

        # ceil_mode variations
        test_cases.extend(
            [
                AveragePoolBackwardTestCase(
                    input_size=(1, 1, 33, 33),
                    kernel_size=4,
                    stride=3,
                    padding=1,
                    ceil_mode=True,
                    pool_dim=2,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(2, 5, 77, 89),
                    kernel_size=(5, 3),
                    stride=(4, 2),
                    padding=(2, 1),
                    ceil_mode=True,
                    pool_dim=2,
                    dtype=dtype,
                ),
            ]
        )

        # ============ 3D Average Pooling Backward Tests ============
        # Basic cubic kernels
        test_cases.extend(
            [
                AveragePoolBackwardTestCase(
                    input_size=(1, 2, 32, 32, 32),
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    pool_dim=3,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(2, 4, 48, 48, 48),
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    pool_dim=3,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(1, 1, 64, 64, 64),
                    kernel_size=7,
                    stride=3,
                    padding=0,
                    pool_dim=3,
                    dtype=dtype,
                ),
            ]
        )

        # Non-cubic kernels
        test_cases.extend(
            [
                AveragePoolBackwardTestCase(
                    input_size=(1, 3, 24, 36, 48),
                    kernel_size=(2, 3, 4),
                    stride=(1, 2, 2),
                    padding=(0, 1, 2),
                    pool_dim=3,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(2, 2, 40, 32, 56),
                    kernel_size=(5, 3, 7),
                    stride=(2, 1, 3),
                    padding=(2, 1, 3),
                    pool_dim=3,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(1, 1, 28, 44, 36),
                    kernel_size=(3, 5, 2),
                    stride=(2, 3, 1),
                    padding=(1, 2, 1),
                    pool_dim=3,
                    dtype=dtype,
                ),
            ]
        )

        # ceil_mode variations
        test_cases.extend(
            [
                AveragePoolBackwardTestCase(
                    input_size=(1, 1, 27, 27, 27),
                    kernel_size=4,
                    stride=3,
                    padding=1,
                    ceil_mode=True,
                    pool_dim=3,
                    dtype=dtype,
                ),
                AveragePoolBackwardTestCase(
                    input_size=(2, 2, 33, 45, 39),
                    kernel_size=(5, 3, 4),
                    stride=(3, 2, 3),
                    padding=(2, 1, 1),
                    ceil_mode=True,
                    pool_dim=3,
                    dtype=dtype,
                ),
            ]
        )

    # Add some edge cases
    edge_cases = [
        # Very large kernels
        AveragePoolBackwardTestCase(
            input_size=(1, 2, 64),
            kernel_size=32,
            stride=16,
            padding=16,
            pool_dim=1,
            dtype=torch.float32,
        ),
        AveragePoolBackwardTestCase(
            input_size=(1, 1, 64, 64),
            kernel_size=32,
            stride=16,
            padding=16,
            pool_dim=2,
            dtype=torch.float16,
        ),
        # Kernel size equals input size
        AveragePoolBackwardTestCase(
            input_size=(1, 2, 16, 16),
            kernel_size=16,
            stride=1,
            padding=8,
            pool_dim=2,
            dtype=torch.bfloat16,
        ),
        # Large stride
        AveragePoolBackwardTestCase(
            input_size=(2, 3, 100, 100),
            kernel_size=5,
            stride=10,
            padding=2,
            pool_dim=2,
            dtype=torch.float32,
        ),
        # Complex 3D case
        AveragePoolBackwardTestCase(
            input_size=(1, 1, 16, 32, 48),
            kernel_size=(8, 4, 6),
            stride=(4, 8, 12),
            padding=(4, 2, 3),
            pool_dim=3,
            dtype=torch.float16,
        ),
    ]

    test_cases.extend(edge_cases)

    print(f"Generated {len(test_cases)} test cases")
    print(f"Data types: {len(dtypes)} types")
    print(f"Pool dimensions: 1D, 2D, 3D")
    print(
        f"Note: AveragePoolBackward test uses default parameters (count_include_pad=True, divisor_override=None)"
    )

    test_writer.add_tests(test_cases)
    test_writer.save()
