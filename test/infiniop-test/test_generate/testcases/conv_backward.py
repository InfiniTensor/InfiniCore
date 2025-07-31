import numpy as np
import gguf
import torch
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def conv_transpose_backward_1d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """1D Conv Transpose Forward + Backward using PyTorch with double precision"""
    # Convert to double precision for computation
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    weight = weight.double().clone().requires_grad_(True)
    if bias is not None:
        bias = bias.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.conv_transpose1d(
        input_tensor,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    # 反向传播
    output.backward(grad_output)

    return (
        output,
        input_tensor.grad,
        weight.grad,
        bias.grad if bias is not None else None,
    )


def conv_transpose_backward_2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    output_padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
):
    """2D Conv Transpose Forward + Backward using PyTorch with double precision"""
    # Convert to double precision for computation
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    weight = weight.double().clone().requires_grad_(True)
    if bias is not None:
        bias = bias.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.conv_transpose2d(
        input_tensor,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    # 反向传播
    output.backward(grad_output)

    return (
        output,
        input_tensor.grad,
        weight.grad,
        bias.grad if bias is not None else None,
    )


def conv_transpose_backward_3d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    output_padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
):
    """3D Conv Transpose Forward + Backward using PyTorch with double precision"""
    # Convert to double precision for computation
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    weight = weight.double().clone().requires_grad_(True)
    if bias is not None:
        bias = bias.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.conv_transpose3d(
        input_tensor,
        weight,
        bias,
        stride=stride,
        padding=padding,
        output_padding=output_padding,
        dilation=dilation,
        groups=groups,
    )

    # 反向传播
    output.backward(grad_output)

    return (
        output,
        input_tensor.grad,
        weight.grad,
        bias.grad if bias is not None else None,
    )


def random_tensor(shape, dtype):
    """Generate random tensor using torch"""
    if dtype == torch.bfloat16:
        tensor = torch.randn(shape, dtype=torch.float32).to(dtype).clamp(-1, 1)
    else:
        tensor = torch.randn(shape, dtype=dtype).clamp(-1, 1)
    return tensor


class ConvTransposeBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        weight_size: Tuple[int, ...],
        bias: bool = True,
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        output_padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        conv_dim: int = 2,  # 1, 2, or 3
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__("conv_transpose_backward")
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.dilation = dilation
        self.groups = groups
        self.conv_dim = conv_dim
        self.has_bias = bias

        # 随机生成输入张量、权重和偏置
        self.input_tensor = random_tensor(input_size, dtype)
        self.weight = random_tensor(weight_size, dtype)
        self.bias = random_tensor((weight_size[1] * groups,), dtype) if bias else None

        # 先进行前向传播获取输出形状
        if self.conv_dim == 1:
            forward_output = F.conv_transpose1d(
                self.input_tensor,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.conv_dim == 2:
            forward_output = F.conv_transpose2d(
                self.input_tensor,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.conv_dim == 3:
            forward_output = F.conv_transpose3d(
                self.input_tensor,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                output_padding=self.output_padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        else:
            raise ValueError(f"Unsupported conv dimension: {self.conv_dim}")

        # 生成随机的grad_output
        self.grad_output = random_tensor(forward_output.shape, dtype)

        # 计算反向梯度
        if self.conv_dim == 1:
            _, self.grad_input, self.grad_weight, self.grad_bias = (
                conv_transpose_backward_1d(
                    self.input_tensor,
                    self.weight,
                    self.bias,
                    self.grad_output,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.dilation,
                    self.groups,
                )
            )
        elif self.conv_dim == 2:
            _, self.grad_input, self.grad_weight, self.grad_bias = (
                conv_transpose_backward_2d(
                    self.input_tensor,
                    self.weight,
                    self.bias,
                    self.grad_output,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.dilation,
                    self.groups,
                )
            )
        elif self.conv_dim == 3:
            _, self.grad_input, self.grad_weight, self.grad_bias = (
                conv_transpose_backward_3d(
                    self.input_tensor,
                    self.weight,
                    self.bias,
                    self.grad_output,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.dilation,
                    self.groups,
                )
            )

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)

        # Define a helper function to convert tensors to numpy and ggml types
        def convert_tensor(tensor):
            if tensor.dtype == torch.bfloat16:
                return (
                    tensor.view(torch.uint16).detach().numpy(),
                    gguf.GGMLQuantizationType.BF16,
                )
            else:
                return tensor.detach().numpy(), np_dtype_to_ggml(
                    tensor.detach().numpy().dtype
                )

        # Add grad_output tensor (input for backward pass) - keep original data type
        grad_output_numpy, ggml_dtype_grad = convert_tensor(self.grad_output)
        test_writer.add_tensor(
            test_writer.gguf_key("grad_output"),
            grad_output_numpy,
            raw_dtype=ggml_dtype_grad,
        )

        # Add input tensor - keep original data type
        input_numpy, ggml_dtype_input = convert_tensor(self.input_tensor)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=ggml_dtype_input,
        )

        # Add weight tensor - keep original data type
        weight_numpy, ggml_dtype_weight = convert_tensor(self.weight)
        test_writer.add_tensor(
            test_writer.gguf_key("weight"),
            weight_numpy,
            raw_dtype=ggml_dtype_weight,
        )

        # Add bias tensor if present - keep original data type
        if self.bias is not None:
            bias_numpy, ggml_dtype_bias = convert_tensor(self.bias)
            test_writer.add_tensor(
                test_writer.gguf_key("bias"),
                bias_numpy,
                raw_dtype=ggml_dtype_bias,
            )

        # Add parameters
        if isinstance(self.stride, int):
            test_writer.add_array(test_writer.gguf_key("stride"), [self.stride])
        else:
            test_writer.add_array(test_writer.gguf_key("stride"), list(self.stride))

        if isinstance(self.padding, int):
            test_writer.add_array(test_writer.gguf_key("padding"), [self.padding])
        else:
            test_writer.add_array(test_writer.gguf_key("padding"), list(self.padding))

        if isinstance(self.dilation, int):
            test_writer.add_array(test_writer.gguf_key("dilation"), [self.dilation])
        else:
            test_writer.add_array(test_writer.gguf_key("dilation"), list(self.dilation))

        if isinstance(self.output_padding, int):
            test_writer.add_array(
                test_writer.gguf_key("output_padding"), [self.output_padding]
            )
        else:
            test_writer.add_array(
                test_writer.gguf_key("output_padding"), list(self.output_padding)
            )

        test_writer.add_int32(test_writer.gguf_key("groups"), self.groups)

        # Add grad_input tensor (output of backward pass) - 使用double精度
        test_writer.add_tensor(
            test_writer.gguf_key("grad_input"),
            self.grad_input.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

        # Add grad_weight tensor - 使用double精度
        test_writer.add_tensor(
            test_writer.gguf_key("grad_weight"),
            self.grad_weight.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )

        # Add grad_bias tensor if present - 使用double精度
        if self.grad_bias is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("grad_bias"),
                self.grad_bias.numpy(),
                raw_dtype=gguf.GGMLQuantizationType.F64,
            )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("conv_transpose_backward.gguf")

    # Data types
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    test_cases = []

    # Generate test cases for each data type and dimension
    for dtype in dtypes:

        # ============ 1D Conv Transpose Backward Tests ============
        # Basic cases
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(2, 4, 8),
                    weight_size=(4, 8, 3),  # (in_channels, out_channels, kernel_size)
                    stride=1,
                    padding=0,
                    conv_dim=1,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(1, 8, 16),
                    weight_size=(8, 16, 5),
                    stride=2,
                    padding=1,
                    conv_dim=1,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(3, 6, 12),
                    weight_size=(6, 12, 7),
                    stride=3,
                    padding=2,
                    output_padding=1,
                    conv_dim=1,
                    dtype=dtype,
                ),
            ]
        )

        # Cases without bias
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(2, 4, 10),
                    weight_size=(4, 6, 3),
                    bias=False,
                    stride=1,
                    padding=1,
                    conv_dim=1,
                    dtype=dtype,
                ),
            ]
        )

        # Dilation cases
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 3, 8),
                    weight_size=(3, 6, 3),
                    stride=1,
                    padding=2,
                    dilation=2,
                    conv_dim=1,
                    dtype=dtype,
                ),
            ]
        )

        # Groups cases
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(2, 4, 12),
                    weight_size=(4, 2, 3),  # out_channels per group = 2
                    groups=2,
                    stride=1,
                    padding=1,
                    conv_dim=1,
                    dtype=dtype,
                ),
            ]
        )

        # ============ 2D Conv Transpose Backward Tests ============
        # Basic cases with square kernels
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(2, 3, 8, 8),
                    weight_size=(3, 6, 3, 3),  # (in_channels, out_channels, H, W)
                    stride=1,
                    padding=0,
                    conv_dim=2,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(1, 8, 16, 16),
                    weight_size=(8, 16, 5, 5),
                    stride=2,
                    padding=1,
                    conv_dim=2,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(2, 6, 12, 12),
                    weight_size=(6, 12, 7, 7),
                    stride=3,
                    padding=2,
                    output_padding=1,
                    conv_dim=2,
                    dtype=dtype,
                ),
            ]
        )

        # Rectangular kernels
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 4, 10, 8),
                    weight_size=(4, 8, 3, 5),
                    stride=(1, 2),
                    padding=(1, 2),
                    conv_dim=2,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(2, 6, 8, 12),
                    weight_size=(6, 12, 5, 3),
                    stride=(2, 1),
                    padding=(2, 1),
                    output_padding=(1, 0),
                    conv_dim=2,
                    dtype=dtype,
                ),
            ]
        )

        # Cases without bias
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 4, 8, 8),
                    weight_size=(4, 8, 3, 3),
                    bias=False,
                    stride=1,
                    padding=1,
                    conv_dim=2,
                    dtype=dtype,
                ),
            ]
        )

        # Dilation cases
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 3, 8, 8),
                    weight_size=(3, 6, 3, 3),
                    stride=1,
                    padding=2,
                    dilation=2,
                    conv_dim=2,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(2, 4, 10, 8),
                    weight_size=(4, 8, 3, 5),
                    stride=1,
                    padding=(3, 4),
                    dilation=(2, 3),
                    conv_dim=2,
                    dtype=dtype,
                ),
            ]
        )

        # Groups cases
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 6, 8, 8),
                    weight_size=(6, 2, 3, 3),  # out_channels per group = 2
                    groups=3,
                    stride=1,
                    padding=1,
                    conv_dim=2,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(2, 8, 10, 10),
                    weight_size=(8, 3, 5, 5),  # out_channels per group = 3
                    groups=2,
                    stride=2,
                    padding=2,
                    conv_dim=2,
                    dtype=dtype,
                ),
            ]
        )

        # ============ 3D Conv Transpose Backward Tests ============
        # Basic cases with cubic kernels
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 2, 4, 4, 4),
                    weight_size=(2, 4, 3, 3, 3),  # (in_channels, out_channels, D, H, W)
                    stride=1,
                    padding=0,
                    conv_dim=3,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(1, 4, 8, 8, 8),
                    weight_size=(4, 8, 5, 5, 5),
                    stride=2,
                    padding=1,
                    conv_dim=3,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(2, 3, 6, 6, 6),
                    weight_size=(3, 6, 7, 7, 7),
                    stride=3,
                    padding=2,
                    output_padding=1,
                    conv_dim=3,
                    dtype=dtype,
                ),
            ]
        )

        # Non-cubic kernels
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 2, 4, 6, 8),
                    weight_size=(2, 4, 3, 3, 5),
                    stride=(1, 1, 2),
                    padding=(1, 1, 2),
                    conv_dim=3,
                    dtype=dtype,
                ),
                ConvTransposeBackwardTestCase(
                    input_size=(1, 3, 6, 4, 8),
                    weight_size=(3, 6, 5, 3, 7),
                    stride=(2, 1, 2),
                    padding=(2, 1, 3),
                    output_padding=(1, 0, 1),
                    conv_dim=3,
                    dtype=dtype,
                ),
            ]
        )

        # Cases without bias
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 2, 4, 4, 4),
                    weight_size=(2, 4, 3, 3, 3),
                    bias=False,
                    stride=1,
                    padding=1,
                    conv_dim=3,
                    dtype=dtype,
                ),
            ]
        )

        # Dilation cases
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 2, 6, 6, 6),
                    weight_size=(2, 4, 3, 3, 3),
                    stride=1,
                    padding=2,
                    dilation=2,
                    conv_dim=3,
                    dtype=dtype,
                ),
            ]
        )

        # Groups cases
        test_cases.extend(
            [
                ConvTransposeBackwardTestCase(
                    input_size=(1, 4, 6, 6, 6),
                    weight_size=(4, 2, 3, 3, 3),  # out_channels per group = 2
                    groups=2,
                    stride=1,
                    padding=1,
                    conv_dim=3,
                    dtype=dtype,
                ),
            ]
        )

    # Add some edge cases
    edge_cases = [
        # Large kernels
        ConvTransposeBackwardTestCase(
            input_size=(1, 2, 8),
            weight_size=(2, 4, 7),
            stride=1,
            padding=3,
            conv_dim=1,
            dtype=torch.float32,
        ),
        ConvTransposeBackwardTestCase(
            input_size=(1, 1, 8, 8),
            weight_size=(1, 2, 7, 7),
            stride=1,
            padding=3,
            conv_dim=2,
            dtype=torch.float16,
        ),
        # Large strides with output padding
        ConvTransposeBackwardTestCase(
            input_size=(1, 3, 4, 4),
            weight_size=(3, 6, 3, 3),
            stride=4,
            padding=1,
            output_padding=3,
            conv_dim=2,
            dtype=torch.bfloat16,
        ),
        # Complex dilation
        ConvTransposeBackwardTestCase(
            input_size=(1, 2, 8, 8),
            weight_size=(2, 4, 3, 3),
            stride=1,
            padding=4,
            dilation=4,
            conv_dim=2,
            dtype=torch.float32,
        ),
        # Large groups
        ConvTransposeBackwardTestCase(
            input_size=(1, 8, 6, 6),
            weight_size=(8, 1, 3, 3),  # Depthwise convolution
            groups=8,
            stride=1,
            padding=1,
            conv_dim=2,
            dtype=torch.float16,
        ),
    ]

    test_cases.extend(edge_cases)

    print(f"Generated {len(test_cases)} test cases")
    print(f"Data types: {len(dtypes)} types")
    print(f"Conv dimensions: 1D, 2D, 3D")

    test_writer.add_tests(test_cases)
    test_writer.save()
