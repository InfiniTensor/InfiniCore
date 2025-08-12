import numpy as np
import gguf
import torch
import torch.nn.functional as F
from typing import List, Union, Tuple, Optional

from .. import InfiniopTestWriter, InfiniopTestCase, np_dtype_to_ggml, gguf_strides


def conv_backward_1d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
    groups: int = 1,
):
    """1D Conv Forward + Backward using PyTorch with double precision"""
    # Convert to double precision for computation
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    weight = weight.double().clone().requires_grad_(True)
    if bias is not None:
        bias = bias.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.conv1d(
        input_tensor,
        weight,
        bias,
        stride=stride,
        padding=padding,
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


def conv_backward_2d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
):
    """2D Conv Forward + Backward using PyTorch with double precision"""
    # Convert to double precision for computation
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    weight = weight.double().clone().requires_grad_(True)
    if bias is not None:
        bias = bias.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.conv2d(
        input_tensor,
        weight,
        bias,
        stride=stride,
        padding=padding,
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


def conv_backward_3d(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    grad_output: torch.Tensor,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int]] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
):
    """3D Conv Forward + Backward using PyTorch with double precision"""
    # Convert to double precision for computation
    input_tensor = input_tensor.double().clone().requires_grad_(True)
    weight = weight.double().clone().requires_grad_(True)
    if bias is not None:
        bias = bias.double().clone().requires_grad_(True)
    grad_output = grad_output.double()

    # 前向传播
    output = F.conv3d(
        input_tensor,
        weight,
        bias,
        stride=stride,
        padding=padding,
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


class ConvBackwardTestCase(InfiniopTestCase):
    def __init__(
        self,
        input_size: Tuple[int, ...],
        weight_size: Tuple[int, ...],
        bias: bool = True,
        stride: Union[int, Tuple] = 1,
        padding: Union[int, Tuple] = 0,
        dilation: Union[int, Tuple] = 1,
        groups: int = 1,
        conv_dim: int = 2,  # 1, 2, or 3
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__("conv_backward")
        self.stride = stride
        self.padding = padding
        # Set output_padding to 0 with correct dimensions
        if conv_dim == 1:
            self.output_padding = 0
        elif conv_dim == 2:
            self.output_padding = (0, 0)
        elif conv_dim == 3:
            self.output_padding = (0, 0, 0)
        self.dilation = dilation
        self.groups = groups
        self.conv_dim = conv_dim
        self.has_bias = bias

        # 随机生成输入张量、权重和偏置
        self.input_tensor = random_tensor(input_size, dtype)
        self.weight = random_tensor(weight_size, dtype)
        self.bias = random_tensor((weight_size[0],), dtype) if bias else None

        # 先进行前向传播获取输出形状
        if self.conv_dim == 1:
            forward_output = F.conv1d(
                self.input_tensor,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.conv_dim == 2:
            forward_output = F.conv2d(
                self.input_tensor,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        elif self.conv_dim == 3:
            forward_output = F.conv3d(
                self.input_tensor,
                self.weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
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
                conv_backward_1d(
                    self.input_tensor,
                    self.weight,
                    self.bias,
                    self.grad_output,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            )
        elif self.conv_dim == 2:
            _, self.grad_input, self.grad_weight, self.grad_bias = (
                conv_backward_2d(
                    self.input_tensor,
                    self.weight,
                    self.bias,
                    self.grad_output,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
            )
        elif self.conv_dim == 3:
            _, self.grad_input, self.grad_weight, self.grad_bias = (
                conv_backward_3d(
                    self.input_tensor,
                    self.weight,
                    self.bias,
                    self.grad_output,
                    self.stride,
                    self.padding,
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
        test_writer.add_array(
            test_writer.gguf_key("grad_output_shape"), 
            list(self.grad_output.shape)
        )

        # Add input tensor - keep original data type
        input_numpy, ggml_dtype_input = convert_tensor(self.input_tensor)
        test_writer.add_tensor(
            test_writer.gguf_key("input"),
            input_numpy,
            raw_dtype=ggml_dtype_input,
        )
        test_writer.add_array(
            test_writer.gguf_key("input_shape"), 
            list(self.input_tensor.shape)
        )

        # Add weight tensor - keep original data type
        weight_numpy, ggml_dtype_weight = convert_tensor(self.weight)
        test_writer.add_tensor(
            test_writer.gguf_key("weight"),
            weight_numpy,
            raw_dtype=ggml_dtype_weight,
        )
        test_writer.add_array(
            test_writer.gguf_key("weight_shape"), 
            list(self.weight.shape)
        )

        # Add bias tensor if present - keep original data type
        if self.bias is not None:
            bias_numpy, ggml_dtype_bias = convert_tensor(self.bias)
            test_writer.add_tensor(
                test_writer.gguf_key("bias"),
                bias_numpy,
                raw_dtype=ggml_dtype_bias,
            )
            test_writer.add_array(
                test_writer.gguf_key("bias_shape"), 
                list(self.bias.shape)
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
        test_writer.add_array(
            test_writer.gguf_key("grad_input_shape"), 
            list(self.grad_input.shape)
        )

        # Add grad_weight tensor - 使用double精度
        test_writer.add_tensor(
            test_writer.gguf_key("grad_weight"),
            self.grad_weight.numpy(),
            raw_dtype=gguf.GGMLQuantizationType.F64,
        )
        test_writer.add_array(
            test_writer.gguf_key("grad_weight_shape"), 
            list(self.grad_weight.shape)
        )

        # Add grad_bias tensor if present - 使用double精度
        if self.grad_bias is not None:
            test_writer.add_tensor(
                test_writer.gguf_key("grad_bias"),
                self.grad_bias.numpy(),
                raw_dtype=gguf.GGMLQuantizationType.F64,
            )
            test_writer.add_array(
                test_writer.gguf_key("grad_bias_shape"), 
                list(self.grad_bias.shape)
            )


def gen_gguf(dtype, filename: str):
    test_writer = InfiniopTestWriter(filename)
    test_cases = []

    # Format: (input_size, weight_size, stride, padding, dilation, groups, conv_dim, bias)
    _TEST_CASES = [
        # 1D Conv Backward Tests
        ((2, 4, 16), (8, 4, 5), 1, 0, 1, 1, 1, True),
        ((2, 4, 32), (8, 4, 3), 2, 1, 1, 1, 1, True),
        ((1, 2, 64), (4, 2, 7), 3, 2, 1, 1, 1, True),
        
        # 2D Conv Backward Tests
        ((2, 3, 16, 16), (6, 3, 3, 3), (1, 2), (0, 1), 2, 1, 2, True),
        ((1, 4, 32, 32), (8, 4, 5, 5), (2, 1), (2, 0), 1, 1, 2, True),
        ((1, 2, 64, 32), (4, 2, 7, 3), (3, 2), (1, 2), 1, 1, 2, True),
        
        # 3D Conv Backward Tests
        ((1, 2, 8, 8, 8), (4, 2, 3, 3, 3), (1, 2, 1), (0, 1, 2), 1, 1, 3, True),
        ((1, 4, 16, 16, 16), (8, 4, 5, 5, 5), (2, 1, 3), (2, 0, 1), 1, 1, 3, True),
        ((1, 2, 32, 16, 8), (4, 2, 7, 3, 5), (3, 2, 1), (1, 2, 0), 1, 1, 3, True),
        
        # Grouped convolution test case
        ((2, 4, 16), (4, 2, 3), 1, 1, 1, 2, 1, True),
    ]

    for input_size, weight_size, stride, padding, dilation, groups, conv_dim, bias in _TEST_CASES:
        test_case = ConvBackwardTestCase(
            input_size=input_size,
            weight_size=weight_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            conv_dim=conv_dim,
            bias=bias,
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
        torch.float32: "conv_backward_float32.gguf",
        torch.float16: "conv_backward_float16.gguf",
        torch.bfloat16: "conv_backward_bfloat16.gguf",
    }

    for dtype in _TENSOR_DTYPES_:
        filename = dtype_filename_map[dtype]
        print(f"Generating {filename} for dtype {dtype}")
        gen_gguf(dtype, filename)
        
    print(f"Generated GGUF files for {len(_TENSOR_DTYPES_)} data types")
    print("Conv dimensions: 1D, 2D, 3D")
    print("Note: ConvBackward test includes grouped convolution and bias cases while contiguous memory is used for all tensors")
